"""Molecular dynamics with TensorFlow 2.11"""

import tensorflow as tf
import numpy as np
from rdkit import Chem
import re

#
##
### FILE READERS
##
#

def readSDFile(file_name, return_atomic_numbers=True):

    suppl = Chem.SDMolSupplier(file_name, removeHs=False)
    molecule = suppl[0]

    num_atoms = molecule.GetNumAtoms()
    periodic_table = {
                        "H": {
                            "atomic_number": 1,
                            "mass": 1.00790,
                        },
                        "C": {
                            "atomic_number": 6,
                            "mass": 12.01070,
                        },
                        "N": {
                            "atomic_number": 7,
                            "mass": 14.00670,
                        },
                        "O": {
                            "atomic_number": 8,
                            "mass": 15.99940,
                        },
                        "F": {
                            "atomic_number": 9,
                            "mass": 18.99840,
                        },
                        "S": {
                            "atomic_number": 16,
                            "mass": 32.06500,
                        },
                        "Cl": {
                            "atomic_number": 17,
                            "mass": 35.45300,
                        },
                        "Br": {
                            "atomic_number": 35,
                            "mass": 79.90400,
                        },
                        "I": {
                            "atomic_number": 53,
                            "mass": 126.90450,
                        },
                    }

    coords = []
    atomic_numbers = []     # or species label
    masses = []
    for atom_idx in range(num_atoms):

        position = molecule.GetConformer().GetAtomPosition(atom_idx)
        coords.append([position.x, position.y, position.z])

        species = molecule.GetAtomWithIdx(atom_idx).GetSymbol()
        if return_atomic_numbers:
            atomic_numbers.append(periodic_table[species]["atomic_number"])
        else:
            atomic_numbers.append(species)

        masses.append(periodic_table[species]["mass"])

    coords = tf.constant(coords)
    atomic_numbers = tf.constant(atomic_numbers)
    masses = tf.constant(masses)

    return num_atoms, coords, atomic_numbers, masses

def readXYZ(file_name):
    """Can also read .forces files."""
    
    xyz = []
    typ = []
    Na = []
    ct = []
    fd = open(file_name, 'r').read()
    rb = re.compile('(\d+?)\n(.*?)\n((?:[A-Z][a-z]?.+?(?:\n|$))+)')
    ra = re.compile(
        '([A-Z][a-z]?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s*?(?:\n|$)'
    )
    s = rb.findall(fd)
    Nc = len(s)
    if Nc == 0:
        raise ValueError('No coordinates found in file. Check formatting of ' + file_name + '.')
    for i in s:
        X = []
        T = []
        ct.append(i[1])
        c = ra.findall(i[2])
        Na.append(len(c))
        for j in c:
            T.append(j[0])
            X.append(j[1])
            X.append(j[2])
            X.append(j[3])
        X = np.array(X, dtype=np.float32)
        X = X.reshape(len(T), 3)
        xyz.append(X)
        typ.append(T)

    return xyz, typ, Na, ct

#
##
### MD UTILS
##
#

class Settings():
    # vv integrator settings
    def __init__(self):
        self.int_type = 'vv_gromacs'
        self.alpha = None
        self.int_vv_gamma = 0.5
        self.dt = 0.001    # time step [ps]
        self.T = 300.
        self.pdbFileName = "mdTraj.pdb"
        self.xyzFileName = "mdTraj.xyz"
        self.forcesFileName = "mdForces.forces"
        self.strideSaving = 1000

class System():

    def __init__(self, chosen_dtype='float32'):

        self.dtype = chosen_dtype

        self.na = 1
        self.masses = 1
        self.charges = None
        self.coords = None
        self.box_sizes = None
        self.dim = 3
    
    # number of atoms, box_sizes vectors, atomic masses, initial coordinates, TF energy function, atomic types
    def getNNSys(self, n, box_sizes, masses, coords, energy_TF, types):
        self.na = n 
        self.dim = 3       
        self.masses = masses
        self.invmasses = tf.constant(1./self.masses, dtype=self.dtype)
        self.box_sizes = box_sizes
        self.coords = coords
        self.energy_tf = energy_TF
        self.types = types
