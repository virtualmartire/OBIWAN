import tensorflow as tf
import math
import numpy as np

from tensorflow import keras
from keras.layers import Input, Dense
from keras.constraints import max_norm

#
##
### AEVComputers
##
#

class WeightedRadialAEVComputer(tf.keras.layers.Layer):

   def __init__(self,
                  radial_AEV_params
               ):
      super(WeightedRadialAEVComputer, self).__init__(name="w_radial_aev_computer")

      self.Rcr = tf.constant(radial_AEV_params["Rcr"], dtype=self.dtype)
      self.EtaR = tf.constant(radial_AEV_params["EtaR"], dtype=self.dtype)
      self.ShfR = tf.constant(radial_AEV_params["ShfR"], dtype=self.dtype)

      self.pairs_params_r = self.doubleCartProd(self.EtaR, self.ShfR)
      self.num_pairs_params_r = tf.shape(self.pairs_params_r)[0]

   @staticmethod
   def doubleCartProd(a, b):
      """Computes all the possible pairs of the coefficients of two 1D vectors (i.e. their cartesian product)."""

      tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0]])  
      tile_a = tf.expand_dims(tile_a, 2) 
      tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1]) 
      tile_b = tf.expand_dims(tile_b, 2) 
      cart = tf.concat([tile_a, tile_b], axis=2)
      cart = tf.reshape(cart, [-1, 2])

      return cart

   @staticmethod
   def f_C(distance, Rc):
      """The original ANAKIN cutoff function."""
      return 0.5 * tf.math.cos(math.pi * distance / Rc) + 0.5     # Assuming all elements in distances are smaller than cutoff

   def GRFormula(self, interesting_distances, interesting_z_j):
      """The ANAKIN AEV radial term."""

      R_ij = tf.expand_dims(interesting_distances, 0)
      z_j = tf.expand_dims(interesting_z_j, 0)
      pairs_params_r = tf.expand_dims(self.pairs_params_r, 1)
      EtaR, ShfR = tf.unstack(pairs_params_r, axis=-1)

      return z_j * tf.math.exp(-EtaR * (R_ij - ShfR)**2) * self.f_C(R_ij, self.Rcr)
   
   @staticmethod
   def intAndSort(boolean_mask):
      int_mask = tf.cast(boolean_mask, tf.int32)
      sorted_int_mask = tf.sort(int_mask, axis=-1, direction='DESCENDING')       # I think that this is slow
      return sorted_int_mask
   
   def getShiftedIndices(self, boolean_mask):
      sorted_int_mask = self.intAndSort(boolean_mask)
      final_indices = tf.where(sorted_int_mask)
      return final_indices
   
   def createFakeRaggedTensor(self, interesting_elements, boolean_mask):
      """Use a boolean_mask to create a fake, ragged tensor actually padded with zeros.
      
      tensor is a 4D tf.Tensor."""

      indices = self.getShiftedIndices(boolean_mask)
      fake_ragged_dim_length = tf.reduce_max( tf.reduce_sum(tf.cast(boolean_mask, tf.int32), axis=-1) )
      all_zero = tf.zeros(shape=(tf.shape(boolean_mask)[0], tf.shape(boolean_mask)[1], fake_ragged_dim_length, tf.shape(interesting_elements)[-1]), dtype=self.dtype)

      fake_ragged_tensor = tf.tensor_scatter_nd_add(all_zero, indices, interesting_elements)

      return fake_ragged_tensor
   
   def call(self, distance_matrices, atomic_numbers_batch):

      num_atoms = tf.shape(atomic_numbers_batch)[-1]
      
      # 1. Preprocess _atomic_numbers_batch_
      atomic_numbers_batch = tf.expand_dims(atomic_numbers_batch, axis=-2)
      z_j_matrix_batch = tf.tile(atomic_numbers_batch, multiples=[1, num_atoms, 1])

      # 2. Compute and apply the cutoff mask
      cutoff_mask = (distance_matrices < self.Rcr) & (distance_matrices != 0.)
      interesting_distances = tf.boolean_mask(distance_matrices, cutoff_mask)
      interesting_z_j = tf.boolean_mask(z_j_matrix_batch, cutoff_mask)

      # 3. Compute the GR addends
      interesting_GR_addends = self.GRFormula(interesting_distances, interesting_z_j)
      interesting_GR_addends = tf.transpose(interesting_GR_addends)

      # 4. Reconstruct the (minimal) shape and sum for the output
      GR_addends = self.createFakeRaggedTensor(interesting_GR_addends, cutoff_mask)     # (batch_size, num_atoms, fake_ragged_dim_length, num_pairs_params_r)
      GR_batch = tf.reduce_sum(GR_addends, axis=-2)

      return GR_batch


class WeightedAngularAEVComputer(tf.keras.layers.Layer):

   def __init__(self,
                  angular_AEV_params,
                  max_molecule_size
               ):
      super(WeightedAngularAEVComputer, self).__init__(name="w_angular_aev_computer")

      self.Rca = tf.constant(angular_AEV_params["Rca"], dtype=self.dtype)
      self.Zeta = tf.constant(angular_AEV_params["Zeta"], dtype=self.dtype)
      self.ShfZ = tf.constant(angular_AEV_params["ShfZ"], dtype=self.dtype)
      self.EtaA = tf.constant(angular_AEV_params["EtaA"], dtype=self.dtype)
      self.ShfA = tf.constant(angular_AEV_params["ShfA"], dtype=self.dtype)

      self.quadruplets_params_a = self.quadrupleCartProd(self.Zeta, self.ShfZ, self.EtaA, self.ShfA)
      self.num_quadruplets_params_a = tf.shape(self.quadruplets_params_a)[0]

      self.precomputed_triplets = self.computeInterestingTriplets(max_molecule_size)
      self.precomputed_pairs = tf.vectorized_map(fn = lambda x: tf.map_fn(self.computePairs, x),
                                                   elems = self.precomputed_triplets)

   @staticmethod    
   def quadrupleCartProd(a, b, c, d):
      """Computes all the possible combinations of four coefficients of four 1D vectors (i.e. their quadruple cartesian product)."""

      tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0]])  
      tile_a = tf.expand_dims(tile_a, 2)

      tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1]) 
      tile_b = tf.expand_dims(tile_b, 2)

      cart = tf.concat([tile_a, tile_b], axis=2) 
      cart = tf.reshape(cart, [-1, 2])

      tile_c = tf.tile(tf.expand_dims(c, 1), [1, tf.shape(cart)[0]])  
      tile_c = tf.expand_dims(tile_c, 2) 
      tile_c = tf.reshape(tile_c, [-1, 1])

      cart = tf.tile(cart, [tf.shape(c)[0], 1])
      cart = tf.concat([cart, tile_c], axis=1)

      tile_d = tf.tile(tf.expand_dims(d, 1), [1, tf.shape(cart)[0]])  
      tile_d = tf.expand_dims(tile_d, 2) 
      tile_d = tf.reshape(tile_d, [-1, 1])

      cart = tf.tile(cart, [tf.shape(d)[0], 1])
      cart = tf.concat([cart, tile_d], axis=1)

      return cart

   @staticmethod
   def computeInterestingTriplets(max_molecule_size):
      """Computes the indices of all the interesting triplets of atoms given the number of atoms in the molecule.
      (`interesting` for the subsequent angles' construction, i.e. without repetition).
      
      This computation is done only one time at the beginning of the session for the biggest molecule of the experiment. All the triplets
      for molecules of other sizes are sliced from the aforesaid big tensor."""
      
      def getPartialCombinations(objects_array):
         
         pairs = tf.meshgrid(objects_array, objects_array)
         pairs = tf.transpose(pairs)
         pairs = tf.reshape(pairs, shape=[-1,2])

         column_1, column_2 = tf.unstack(pairs, axis=1)
         interesting_pairs = tf.boolean_mask(pairs, column_1 < column_2)

         return interesting_pairs

      # First of all compute the triplets

      atoms_indexes = tf.range(max_molecule_size, dtype=tf.int64)

      atoms_indexes_ = tf.expand_dims(atoms_indexes, 0)
      atoms_indexes_ = tf.repeat(atoms_indexes_, max_molecule_size, axis=0)
      non_diagonal_selector = tf.ones([max_molecule_size, max_molecule_size]) - tf.eye(max_molecule_size)
      atoms_indexes_ = tf.boolean_mask(atoms_indexes_, non_diagonal_selector)
      atoms_indexes_ = tf.reshape(atoms_indexes_, shape=[max_molecule_size, max_molecule_size-1])

      interesting_pairs = tf.map_fn(getPartialCombinations, atoms_indexes_)

      num_triplets = (max_molecule_size-1)*((max_molecule_size-1)-1)//2
      atoms_indexes = tf.expand_dims(tf.reshape(tf.repeat(atoms_indexes, num_triplets),
                                                shape=[max_molecule_size, num_triplets]),
                                    -1)
      interesting_triplets = tf.concat([atoms_indexes, interesting_pairs], -1)

      # And then reorder the triplets in order to fetch them with ease

      max_triplets_per_atom = tf.shape(interesting_triplets)[1]
      current_triplets_per_atom = ((max_molecule_size)-1) * ((max_molecule_size)-2) // 2
      indices = tf.stack([tf.range(max_triplets_per_atom),
                           tf.repeat([2], repeats=max_triplets_per_atom)],
                           axis=-1)
      matrix = tf.squeeze(tf.gather(interesting_triplets, [0]))
      last_coeff = tf.gather_nd(indices=indices, params=matrix)
      sorted_last_coeff = tf.argsort(last_coeff)
      selected_indices = tf.gather(sorted_last_coeff, tf.range(current_triplets_per_atom))
      second_indices = tf.repeat(tf.expand_dims(selected_indices, axis=0),
                                 repeats=max_molecule_size,
                                 axis=0)
      second_indices = tf.reshape(second_indices, [-1])
      first_indices = tf.repeat(tf.range(max_molecule_size), repeats=current_triplets_per_atom)
      indices = tf.stack([first_indices, second_indices], axis=1)
      output = tf.gather_nd(params=interesting_triplets, indices=indices)
      output = tf.reshape(output, [max_molecule_size, current_triplets_per_atom, 3])

      return output

   @staticmethod
   def computePairs(triplet_indexes):
      """Given three triplet's indices, compute all the 3 possible pairs."""
      pairs = tf.meshgrid(triplet_indexes, triplet_indexes)
      pairs = tf.transpose(pairs)
      pairs = tf.reshape(pairs, shape=[-1,2])

      return tf.gather(pairs, [1, 2, 5])

   def getInterestingTriplets(self, current_molecule_size):
      current_triplets_per_atom = ((current_molecule_size)-1) * ((current_molecule_size)-2) // 2
      return tf.slice(self.precomputed_triplets, [0, 0, 0], [current_molecule_size, current_triplets_per_atom, 3])

   def getPairs(self, current_molecule_size):
      current_triplets_per_atom = ((current_molecule_size)-1) * ((current_molecule_size)-2) // 2
      return tf.slice(self.precomputed_pairs, [0, 0, 0, 0], [current_molecule_size, current_triplets_per_atom, 3, 2])

   @staticmethod
   def carnotTheorem(distances):
      """Implementation of the so called Carnot Formula in order to compute angles from distances (which we already have!)."""

      R_ij, R_ik, R_jk = tf.unstack(distances, axis=-1)
      cos_alpha = (R_ij**2 + R_ik**2 - R_jk**2) / tf.clip_by_value(2*R_ij*R_ik, clip_value_min=1e-10, clip_value_max=np.inf)
      alpha = tf.math.acos(0.95*cos_alpha)

      return tf.stack([alpha, R_ij, R_ik], axis=1)    # returns also the interesting distances

   @staticmethod
   def f_C(distance, Rc):
      """The original ANAKIN cutoff function."""
      return 0.5 * tf.math.cos(math.pi * distance / Rc) + 0.5     # Assuming all elements in distances are smaller than cutoff

   def GAFormula(self, interesting_geometries, interesting_species_jk):
         """The ANAKIN AEV angular term."""

         interesting_geometries = tf.expand_dims(interesting_geometries, 0)
         interesting_species_jk = tf.expand_dims(interesting_species_jk, 0)
         quadruplets_params_a = tf.expand_dims(self.quadruplets_params_a, 1)
         angle, R_ij, R_ik = tf.unstack(interesting_geometries, axis=-1)
         z_j, z_k = tf.unstack(interesting_species_jk, axis=-1)
         Zeta, ShfZ, EtaA, ShfA = tf.unstack(quadruplets_params_a, axis=-1)

         factor_1 = ( (1 + tf.math.cos(angle - ShfZ)) / 2 ) ** Zeta
         factor_2 = tf.math.exp( -EtaA * ( (R_ij+R_ik)/2 - ShfA )**2 )
         GA_addends = z_j*z_k * 2 * factor_1 * factor_2 * self.f_C(R_ij, self.Rca) * self.f_C(R_ik, self.Rca)

         return GA_addends

   @staticmethod
   def intAndSort(boolean_mask):
      int_mask = tf.cast(boolean_mask, tf.int32)
      sorted_int_mask = tf.sort(int_mask, axis=-1, direction='DESCENDING')       # I think that this is slow
      return sorted_int_mask
   
   def getShiftedIndices(self, boolean_mask):
      sorted_int_mask = self.intAndSort(boolean_mask)
      final_indices = tf.where(sorted_int_mask)
      return final_indices
   
   def createFakeRaggedTensor(self, interesting_elements, boolean_mask):
      """Use a boolean_mask to create a fake, ragged tensor actually padded with zeros.
      
      tensor is a 4D tf.Tensor."""

      indices = self.getShiftedIndices(boolean_mask)
      fake_ragged_dim_length = tf.reduce_max( tf.reduce_sum(tf.cast(boolean_mask, tf.int32), axis=-1) )
      all_zero = tf.zeros(shape=(tf.shape(boolean_mask)[0], tf.shape(boolean_mask)[1], fake_ragged_dim_length, tf.shape(interesting_elements)[-1]), dtype=self.dtype)

      fake_ragged_tensor = tf.tensor_scatter_nd_add(all_zero, indices, interesting_elements)

      return fake_ragged_tensor

   def call(self, distance_matrices, atomic_numbers_batch):

      # 1. Set some environment variables
      batch_size = tf.shape(atomic_numbers_batch)[0]
      num_atoms = tf.shape(atomic_numbers_batch)[1]

      # 2. Get the distances' and species' triplets
      distance_indices = tf.tile(tf.expand_dims(self.getPairs(num_atoms), axis=0),
                                 multiples=[batch_size, 1, 1, 1, 1])
      triplets_distances = tf.gather_nd(params=distance_matrices, indices=distance_indices, batch_dims=1)   # we need all the three distances in order to compute the angles
      species_indices = tf.tile(tf.expand_dims(self.getInterestingTriplets(num_atoms), axis=0),
                                 multiples=[batch_size, 1, 1, 1])
      triplets_species = tf.gather(params=atomic_numbers_batch, indices=species_indices, batch_dims=1)

      # 3. Compute and apply the cutoff mask
      cutoff_mask = tf.reduce_all(triplets_distances < [self.Rca, self.Rca, 1e20], axis=-1) & tf.reduce_all(triplets_distances != [0., 0., 0.], axis=-1)
      interesting_distances = tf.boolean_mask(triplets_distances, cutoff_mask)
      interesting_species = tf.boolean_mask(triplets_species, cutoff_mask)

      # 4. Preprocess the inputs for the self.GAFormula
      interesting_geometries = self.carnotTheorem(interesting_distances)
      interesting_species_jk = tf.slice(interesting_species, begin=[0, 1], size=[-1, 2])

      # 5. Compute the GA addends
      interesting_GA_addends = self.GAFormula(interesting_geometries, interesting_species_jk)
      interesting_GA_addends = tf.transpose(interesting_GA_addends)

      # 6. Reconstruct the (minimal) shape and sum for the output
      GA_addends = self.createFakeRaggedTensor(interesting_GA_addends, cutoff_mask)         # (batch_size, num_atoms, fake_ragged_dim_length, num_quadruplets_params_a)
      GA_batch = tf.reduce_sum(GA_addends, axis=-2)

      return GA_batch

#
##
### NON-LEARNABLE LAYERS
##
#

class CoordToDist(tf.keras.layers.Layer):

   def __init__(self):
      super(CoordToDist, self).__init__()

   def call(self, coordinates_batch):
      t1 = tf.expand_dims(coordinates_batch, axis=1)
      t2 = tf.expand_dims(coordinates_batch, axis=2)
      return tf.norm(t1 - t2, ord = 'euclidean', axis = 3)

class CELU(tf.keras.layers.Layer):
   
   def __init__(self, alpha):
      super(CELU, self).__init__()
      self.alpha = alpha

   def call(self, x):
      safe_x = tf.where(x>0., 0., x)      # due to a TensorFlow bug
      return tf.where(x>0., x, self.alpha * (tf.math.exp(safe_x/self.alpha) - 1))

#
##
### MODELS
##
#

class WAnakinOE(tf.keras.Model):
   """A version of WAnakin with the Anakin's ACSF optimized for off-equilibrium systems."""

   def __init__(self, max_molecule_size):
      super(WAnakinOE, self).__init__(name="w_anakin_oe")

      # torchani-2x hyperparameters
      radial_AEV_params = {
                           "Rcr": 5.1000e+00,
                           "EtaR": [1.9700000e+01],
                           "ShfR": [8.0000000e-01,1.0687500e+00,1.3375000e+00,1.6062500e+00,1.8750000e+00,2.1437500e+00,2.4125000e+00,2.6812500e+00,2.9500000e+00,3.2187500e+00,3.4875000e+00,3.7562500e+00,4.0250000e+00,4.2937500e+00,4.5625000e+00,4.8312500e+00]
                        }
      angular_AEV_params = {
                              "Rca": 3.5000e+00,
                              "Zeta": [1.4100000e+01],
                              "ShfZ": [3.9269908e-01,1.1780972e+00,1.9634954e+00,2.7488936e+00],
                              "EtaA": [1.2500000e+01],
                              "ShfA": [8.0000000e-01,1.1375000e+00,1.4750000e+00,1.8125000e+00,2.1500000e+00,2.4875000e+00,2.8250000e+00,3.1625000e+00]
                           }
      vocabulary = ["H", "C", "N", "O",
                     "F", "S", "Cl"]
      self_energies_list = [-0.5978583943827134, -38.08933878049795, -54.711968298621066, -75.19106774742086,
                              -99.80348506781634, -398.1577125334925, -460.1681939421027]      # torchani-2x self-energies (matched with the vocabulary)

      # Input manipulators
      self.distances_computer = CoordToDist()
      self.species_translator = tf.keras.layers.StringLookup(vocabulary=vocabulary, output_mode='int', mask_token='', num_oov_indices=0)

      # AEV computers
      periodic_table = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl']
      self.atomic_numbers_converter = tf.keras.layers.StringLookup(vocabulary=periodic_table, output_mode='int', mask_token='', num_oov_indices=0)
      self.radial_aev_computer = WeightedRadialAEVComputer(radial_AEV_params)
      self.angular_aev_computer = WeightedAngularAEVComputer(angular_AEV_params, max_molecule_size)

      # Energy MLPs
      radial_aev_lenght = 1
      for key in radial_AEV_params.keys():
         if key != 'Rcr':
            radial_aev_lenght *= len(radial_AEV_params[key])
      angular_aev_lenght = 1
      for key in angular_AEV_params.keys():
         if key != 'Rca':
            angular_aev_lenght *= len(angular_AEV_params[key])
      self.aev_lenght = int(radial_aev_lenght + angular_aev_lenght)
      self.networks = [self.getMLP([256, 192, 160]),
                        self.getMLP([224, 192, 160]),
                        self.getMLP([192, 160, 128]),
                        self.getMLP([192, 160, 128]),
                        self.getMLP([160, 128, 96]),
                        self.getMLP([160, 128, 96]),
                        self.getMLP([160, 128, 96])]   # torchani-2x MLPs architectures (matched with the vocabulary)

      # Self energies (we used this anakin mechanism instead of "normalizing" the target)
      self.self_energies = tf.constant(self_energies_list)

   def getMLP(self, neurons_per_hidden_layer):

      inputs = Input(shape=(self.aev_lenght))

      x = Dense(neurons_per_hidden_layer[0], kernel_constraint=max_norm(3.), kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
      x = CELU(0.1)(x)
      for neurons in neurons_per_hidden_layer[1:]:
         x = Dense(neurons, kernel_constraint=max_norm(3.), kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
         x = CELU(0.1)(x)
      
      outputs = Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)

      return keras.Model(inputs=inputs, outputs=outputs)

   def loadWeights(self, model_path):
      self.build( [(None, None, 3), (None, None)] )
      model_for_weights = tf.keras.models.load_model(model_path)
      self.set_weights(model_for_weights.get_weights())

   def computeMLPEnergies(self, masked_aevs, masked_num_species):

      masked_mlp_energies = tf.zeros_like(masked_num_species, dtype=self.dtype)

      for i, mlp in enumerate(self.networks):
         indices = tf.where(masked_num_species == i+1)
         masked_mlp_energies = tf.cond(pred = tf.shape(indices)[0] > 0,
                                       true_fn = lambda: tf.tensor_scatter_nd_add(
                                                                              masked_mlp_energies,
                                                                              indices, 
                                                                              tf.reshape( mlp(tf.reshape( tf.gather(masked_aevs, indices), [-1, self.aev_lenght] )), [-1])    # reshape due to indices.shape
                                                                                 ),
                                       false_fn = lambda: masked_mlp_energies)

      return masked_mlp_energies

   def call(self, inputs):
      """Compute the molecular energies from the input coordinates and species."""

      coordinates_batch, string_species_batch = inputs
      batch_size = tf.shape(coordinates_batch)[0]
      num_atoms = tf.shape(coordinates_batch)[1]

      # 1. Preprocess the inputs
      distance_matrices = self.distances_computer(coordinates_batch)
      num_species_batch = tf.cast(self.species_translator(string_species_batch), dtype=self.dtype)    # chemical elements counted as they appear in _vocabulary_
      dummy_mask = num_species_batch != 0.     # (batch_size, num_atoms)
      atomic_numbers_batch = tf.cast(self.atomic_numbers_converter(string_species_batch), dtype=self.dtype)

      # 2. Compute, concatenate and "normalize" the AEVs
      rad_aevs_batch = self.radial_aev_computer(distance_matrices, atomic_numbers_batch)
      ang_aevs_batch = self.angular_aev_computer(distance_matrices, atomic_numbers_batch)
      aevs_batch = tf.concat([rad_aevs_batch, ang_aevs_batch], axis=-1)
      aevs_batch = (aevs_batch - tf.math.reduce_mean(aevs_batch, axis=-1, keepdims=True)) / (tf.math.reduce_std(aevs_batch, axis=-1, keepdims=True) + tf.keras.backend.epsilon())

      # 3. Compute the per-atom self and MLP energies
      masked_num_species = tf.cast(tf.boolean_mask(num_species_batch, dummy_mask), dtype=tf.int32)    # (nondummy_atoms,)
      masked_aevs = tf.boolean_mask(aevs_batch, dummy_mask)
      masked_mlp_energies = self.computeMLPEnergies(masked_aevs, masked_num_species)
      masked_self_energies = tf.gather(self.self_energies, masked_num_species-1)

      # 4. Reconstruct the shape and sum to obtain the molecular energies
      masked_complete_atomic_energies = masked_self_energies + masked_mlp_energies
      complete_atomic_energies = tf.zeros([batch_size, num_atoms], dtype=self.dtype)
      complete_atomic_energies = tf.tensor_scatter_nd_add(complete_atomic_energies, tf.where(dummy_mask), masked_complete_atomic_energies)
      molecular_energies = tf.reduce_sum(complete_atomic_energies, -1)

      return molecular_energies