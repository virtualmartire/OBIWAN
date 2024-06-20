"""Molecular dynamics with TensorFlow 2.13"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
import gc

# basic molecular dynamics code
# langevin (a la gromacs https://manual.gromacs.org/current/reference-manual/algorithms/stochastic-dynamics.html) integration method 
class MolecularDynamics():    

    def __init__(self, settings, system, chosen_dtype='float32', wrapping=True, save_forces=False, use_atomic_numbers=True):

        self.dtype = chosen_dtype

        self.E_nn = 0
        # self.storePositionsAndPot = False
        self.k_boltz =  tf.constant(0.00831446261815324, dtype=self.dtype)        # [kJ/(mol*K)]
        self.C1 = 47.992430                                                         # C1 = h/k_boltz = 47.992430 [ps*K] (assuming nu is in [ps]^-1)
        self.setSystem(system)
        self.setSettings(settings)
        self.wrapping = wrapping
        self.save_forces = save_forces
        self.use_atomic_numbers = use_atomic_numbers

        self.typeDict = {6:'C',7:'N',1:'H',8:'O',9:'F',17:'Cl',35:'Br',53:'I',16:'S'}

    def setSystem(self, system):
        self.system = system
        self.dim = system.dim
        self.na = tf.shape(system.coords)[0]        # excluding tha batch axis
        self.pos = tf.identity(self.system.coords)
        self.hvel = tf.zeros_like(self.pos)
        self.step = 0
        self.masses = self.system.masses
        self.invmasses = 1./self.masses
        self.box_sizes = tf.identity(self.system.box_sizes)                
        self.half_box_sizes = self.box_sizes/2.
        self.types = tf.identity(system.types)
        
    def setSettings(self, settings):
        self.settings = settings    
        tf.random.set_seed(self.settings.seed)
        self.setIntegrator()

    def integratorVV(self):
    
        rg = tf.random.normal((self.na, self.dim), dtype=self.dtype)
        self.v = self.hvel + (self.invmasses*self.forces*self.dt)
        dv = -self.int_vv_alpha*self.v+self.int_vv_const*rg
        self.pos = self.pos+(self.v+0.5*dv)*self.dt    
        self.hvel = self.v+dv

    def setIntegrator(self):        
        self.dt = self.settings.dt
        if (self.settings.int_type == 'vv_gromacs'):                        
            print("<<INFO>> VV gamma %f ps^-1"%self.settings.int_vv_gamma)            
            self.int_vv_alpha = tf.cast(1.-tf.exp(-self.settings.int_vv_gamma*self.dt), dtype=self.dtype)
            self.int_vv_const = tf.sqrt(self.k_boltz*self.settings.T*self.invmasses*self.int_vv_alpha*(2.-self.int_vv_alpha))                        
            self.integrator = self.integratorVV
        else:
            print("<<ERROR>> Unknow integrator!")
            exit(-1)

    def openPDBFile(self):                        
        self.PDB_FILE = open(self.settings.pdbFileName, "w")
        if (self.system.dim==3):
            # write box sizes in Angstrom
            self.PDB_FILE.write("CRYST1 %8.3f %8.3f %8.3f  90.00  90.00  90.00\n"%(self.box_sizes[0]*10., self.box_sizes[1]*10., self.box_sizes[2]*10.))

    def savePDBFrame(self):
        
        # transfer data to host mem 
        pos_host = self.pos_wrapped.numpy()
        
        self.PDB_FILE.write("MODEL\n")     
        stru = ""
        for i in range(0,self.na):
            type = self.typeDict[int(self.types[i])]
            # save positions in Angostrom
            stri = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format("ATOM",i+1,type,"","XXX","A",1,"",pos_host[i,0]*10., pos_host[i,1]*10., pos_host[i,2]*10.,1.,1.)
            stru = stru+stri
        self.PDB_FILE.write(stru)       
        self.PDB_FILE.write("END MODEL\n")

    def closePDBFile(self):
        self.PDB_FILE.close()

    def openXYZFile(self, append_results):
        if append_results:
            self.XYZ_FILE = open(self.settings.xyzFileName, "a")
        else:
            self.XYZ_FILE = open(self.settings.xyzFileName, "w")

    def saveXYZFrame(self):
        
        # transfer data to host mem 
        pos_host = self.pos_wrapped.numpy()
        
        # write
        self.XYZ_FILE.write(f"{self.na}\n")
        self.XYZ_FILE.write(f"{self.E_nn}\n")
        for i in range(0, self.na):
            if self.use_atomic_numbers:
                type = self.typeDict[int(self.types[i])]
            else:
                type = self.types[i].numpy().decode('utf-8')
            # save positions in Angostrom
            self.XYZ_FILE.write(f"{type} {pos_host[i,0]*10.:.3f} {pos_host[i,1]*10.:.3f} {pos_host[i,2]*10.:.3f}\n")

    def closeXYZFile(self):
        self.XYZ_FILE.close()

    def openForcesFile(self, append_results):
        if append_results:
            self.FORCES_FILE = open(self.settings.forcesFileName, "a")
        else:
            self.FORCES_FILE = open(self.settings.forcesFileName, "w")

    def saveForcesFrame(self):
        
        # transfer data to host mem 
        forces_host = self.forces.numpy()
        
        # write
        self.FORCES_FILE.write(f"{self.na}\n")
        self.FORCES_FILE.write(f"{self.E_nn}\n")
        for i in range(0, self.na):
            if self.use_atomic_numbers:
                type = self.typeDict[int(self.types[i])]
            else:
                type = self.types[i].numpy().decode('utf-8')
            self.FORCES_FILE.write(f"{type} {forces_host[i,0]*10.:.3f} {forces_host[i,1]*10.:.3f} {forces_host[i,2]*10.:.3f}\n")

    def closeForcesFile(self):
        self.FORCES_FILE.close()

    @staticmethod
    def centerSystem(pos):
        """Center the system around the origin."""
        means = tf.reduce_mean(pos, axis=0, keepdims=True)
        return pos - means

    def initState(self):
        self.forces = tf.zeros_like(self.pos)
        self.hvel = tf.zeros_like(self.pos)
        self.pos = self.centerSystem(self.system.coords)

    def computeForcesAndEnergies(self):                       
        with tf.GradientTape(watch_accessed_variables=False) as force_tape:
            force_tape.watch(self.pos)
            self.E_nn = self.system.energy_tf(self.pos, self.box_sizes, self.types)
        self.forces = -force_tape.gradient(self.E_nn, self.pos)

    # minimize the current system through gradient ascent      
    def minimize(self, eta=1e-5, stop_criterion=['max_force', 1e-6], maxIt=100):

        stop_criterion_kind = stop_criterion[0]

        if stop_criterion_kind == 'max_force':
            wanted_max_force = stop_criterion[1]
            max_force = 1e3
            it = 0
            while(max_force > wanted_max_force and it < maxIt):
                self.computeForcesAndEnergies()      # re-assign self.forces and self.E_nn
                # now moves on the forces direction (gradient ascent)
                self.pos = self.pos + eta*self.forces
                max_force = tf.reduce_max(tf.abs(self.forces))
                if it % 1000 == 0:
                    print(f"<<INFO>> Minimization: It {it} Potential value {self.E_nn} max force value {max_force}")
                
                if it % 1000 == 0:
                    # garbage collection
                    tf.keras.backend.clear_session()
                    _ = gc.collect()

                it += 1
            print(f"<<INFO>> Minimization: It {it} Potential value {self.E_nn} max force value {max_force}")

        else:
            raise ValueError("Stop criterion not implemented.")

    def wrap(self):

        # how many half_box_sizes we are far from the origin?
        quotient = tf.cast(self.pos/self.half_box_sizes, tf.int32)

        # but transl only every 2 half_box_sizes
        quotient = tf.cast(quotient, self.dtype)
        transl_factor = tf.math.sign(quotient) * tf.math.ceil( tf.abs(quotient/2.) )
        self.pos_wrapped = self.pos - transl_factor*self.box_sizes

    # pass the num of steps and the custom potentials parameters (if any)
    def run(self, numsteps, append_results=False):

        self.relativeStep = 0
        print(f"<<INFO>> Total sim time: {(float(numsteps)*self.settings.dt)/1000.} ns")
        print(f"<<INFO>> Time step: {self.settings.dt} ps")

        #self.openPDBFile()
        self.openXYZFile(append_results)
        if self.save_forces:
            self.openForcesFile(append_results)

        print("<<INFO>> Start looping..")
        pbar = tqdm(total=numsteps)
        stride = np.min([1000,numsteps])

        for step in (range(0, numsteps)):

            # start time postponed to get net time excluding graph compilation
            if (self.relativeStep==1):
                startTime = time.perf_counter()

            # compute forces
            self.computeForcesAndEnergies()
            
            # integrate (i.e. update positions and velocities)
            self.integrator()            
            
            if (step % (numsteps//stride) == 0):
                pbar.update(numsteps//stride)
            
            # save trajectory and energy (after wrapping!)
            if (step % self.settings.strideSaving == 0 and self.system.dim==3):
                if self.wrapping:
                    self.wrap()
                else:       # for saving
                    self.pos_wrapped = self.pos
                #self.savePDBFrame()
                self.saveXYZFrame()
                if self.save_forces:
                    self.saveForcesFrame()

            if step % 1000 == 0:
                # garbage collection
                tf.keras.backend.clear_session()
                _ = gc.collect()

            self.step += 1
            self.relativeStep +=1 

        pbar.close()
        #self.closePDBFile()
        self.closeXYZFile()
        if self.save_forces:
            self.closeForcesFile()
        
        # print final stats
        endTime = time.perf_counter()
        ns = numsteps*self.dt*1e-3
        dtime = (endTime-startTime)/3600./24.
        perf = ns/dtime
        print("<<INFO>> Performing %.2f ns/day"%perf)
