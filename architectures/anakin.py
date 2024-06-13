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

class RadialAEVComputer(tf.keras.layers.Layer):

   def __init__(self,
                  vocabulary,
                  radial_AEV_params
               ):
      super(RadialAEVComputer, self).__init__(name="radial_aev_computer")

      self.vocabulary = range(1, len(vocabulary)+1)
      
      self.cutoff = tf.constant(radial_AEV_params["Rcr"], dtype=self.dtype)
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

   def computeGRTerms(self, interesting_distances, pairs_params_r):
      """The ANAKIN AEV radial term."""

      R_ij = tf.expand_dims(interesting_distances, 0)
      pairs_params_r = tf.expand_dims(pairs_params_r, 1)
      EtaR, ShfR = tf.unstack(pairs_params_r, axis=-1)

      return tf.math.exp(-EtaR * (R_ij - ShfR)**2) * self.f_C(R_ij, self.cutoff)
   
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
      all_zero = tf.zeros(shape=(tf.shape(boolean_mask)[0], tf.shape(boolean_mask)[1], fake_ragged_dim_length, len(self.vocabulary), tf.shape(interesting_elements)[-1]), dtype=self.dtype)

      fake_ragged_tensor = tf.tensor_scatter_nd_add(all_zero, indices, interesting_elements)

      return fake_ragged_tensor
   
   def tailorSpeciesBatch(self, num_species_batch, cutoff_mask):

      num_atoms = tf.shape(num_species_batch)[1]

      interesting_species = tf.expand_dims(num_species_batch, axis=1)
      interesting_species = tf.tile(interesting_species, multiples=[1, num_atoms, 1])
      interesting_species = tf.boolean_mask(interesting_species, cutoff_mask)

      return interesting_species

   def call(self, distance_matrices_batch, num_species_batch):

      # 1. Compute and apply the cutoff mask
      cutoff_mask = (distance_matrices_batch < self.cutoff) & (distance_matrices_batch != 0.)      # != 0. to exclude dummy atoms, that are all in the same place
      interesting_distances = tf.boolean_mask(distance_matrices_batch, cutoff_mask)
      interesting_species = self.tailorSpeciesBatch(num_species_batch, cutoff_mask)

      # 2. Compute the GR addends
      interesting_GR_addends = self.computeGRTerms(interesting_distances, self.pairs_params_r)
      interesting_GR_addends = tf.transpose(interesting_GR_addends)

      # 3. Reconstruct the (minimal) shape and sum for the output
      interesting_species_one_hot = tf.one_hot(
                                          tf.cast(interesting_species, tf.int32)-1,    # -1 to one-hot H in the first component
                                          len(self.vocabulary)
                                       )
      interesting_species_one_hot = tf.expand_dims(interesting_species_one_hot, axis=-1)
      interesting_GR_addends = tf.expand_dims(interesting_GR_addends, axis=1)
      interesting_GR_addends = tf.tile(interesting_GR_addends, multiples=[1, len(self.vocabulary), 1])
      GR_addends = interesting_GR_addends * tf.cast(interesting_species_one_hot, interesting_GR_addends.dtype)
      GR_addends = self.createFakeRaggedTensor(GR_addends, cutoff_mask)          # (batch_size, num_atoms, fake_ragged_dim_length, vocabulary_length, num_pairs_params_r)
      GR_batch = tf.reduce_sum(GR_addends, axis=2)
      GR_batch = tf.reshape(GR_batch, [tf.shape(GR_batch)[0], tf.shape(GR_batch)[1], -1])

      return GR_batch


class AngularAEVComputer(tf.keras.layers.Layer):

   def __init__(self,
                  vocabulary,
                  angular_AEV_params,
               ):
      super(AngularAEVComputer, self).__init__(name="angular_aev_computer")

      num_elements = len(vocabulary)
      self.pairs_vocabulary = []
      for i, x in enumerate( range(1, num_elements+1) ):
         for y in range(1, num_elements+1)[i:]:
            self.pairs_vocabulary.append([x, y])

      self.cutoff = tf.constant(angular_AEV_params["Rca"], dtype=self.dtype)
      self.Zeta = tf.constant(angular_AEV_params["Zeta"], dtype=self.dtype)
      self.ShfZ = tf.constant(angular_AEV_params["ShfZ"], dtype=self.dtype)
      self.EtaA = tf.constant(angular_AEV_params["EtaA"], dtype=self.dtype)
      self.ShfA = tf.constant(angular_AEV_params["ShfA"], dtype=self.dtype)

      self.quadruplets_params_a = self.quadrupleCartProd(self.Zeta, self.ShfZ, self.EtaA, self.ShfA)
      self.num_quadruplets_params_a = tf.shape(self.quadruplets_params_a)[0]

      self.num_pairs = len(self.pairs_vocabulary)

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

   def computeGATerms(self, interesting_geometries, quadruplets_params_a):
         """The ANAKIN AEV angular term."""
         
         interesting_geometries = tf.expand_dims(interesting_geometries, 0)
         quadruplets_params_a = tf.expand_dims(quadruplets_params_a, 1)
         angle, R_ij, R_ik = tf.unstack(interesting_geometries, axis=-1)
         Zeta, ShfZ, EtaA, ShfA = tf.unstack(quadruplets_params_a, axis=-1)

         factor_1 = ( (1 + tf.math.cos(angle - ShfZ)) / 2 ) ** Zeta
         factor_2 = tf.math.exp( -EtaA * ( (R_ij+R_ik)/2 - ShfA )**2 )
         GA_addends = 2 * factor_1 * factor_2 * self.f_C(R_ij, self.cutoff) * self.f_C(R_ik, self.cutoff)

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

   @staticmethod
   def computeSpeciesTripletsIndices(flattened_triplets_indices):

      flattened_species_indices = tf.slice(flattened_triplets_indices, [0, 0, 0], [tf.shape(flattened_triplets_indices)[0], 2, 3])
      batch_indices, first_indices, second_indices = tf.unstack(flattened_species_indices, axis=-1)
      flattened_species_indices = tf.stack([batch_indices, second_indices], axis=-1)
      central_species_indices = tf.stack([batch_indices, first_indices], axis=-1)
      central_species_indices = tf.slice(central_species_indices, [0, 0, 0], [tf.shape(central_species_indices)[0], 1, 2])
      flattened_species_indices = tf.concat([central_species_indices, flattened_species_indices], axis=1)

      return flattened_species_indices
   
   @staticmethod
   def computeNeighboursCombinations(num_neighbors):

      objects_array = tf.range(num_neighbors, dtype=tf.int32)
      pairs = tf.meshgrid(objects_array, objects_array)
      pairs = tf.transpose(pairs)
      pairs = tf.reshape(pairs, shape=[-1,2])
      column_1, column_2 = tf.unstack(pairs, axis=1)
      interesting_pairs = tf.boolean_mask(pairs, column_1 < column_2)

      return interesting_pairs

   def computeDistancesTripletsIndices(self, cutoff_mask):

      neighbours_count = tf.reduce_sum(tf.cast(cutoff_mask, tf.int32), axis=-1)
      neighbours_count = tf.reshape(neighbours_count, shape=[-1])
      index_shifts = tf.cumsum(neighbours_count, exclusive=True)

      max_neighbors = tf.reduce_max(neighbours_count)
      neighbours_pairs_combinations = self.computeNeighboursCombinations(max_neighbors)

      neighbours_pairs_combinations = tf.tile(neighbours_pairs_combinations, multiples=[tf.shape(neighbours_count)[0], 1])
      max_triplets = max_neighbors*(max_neighbors-1)//2
      neighbours_pairs_combinations = tf.reshape(neighbours_pairs_combinations, shape=[-1, max_triplets, 2])

      permutations_mask = tf.reshape(neighbours_count, [-1, 1, 1])
      permutations_mask = tf.tile(permutations_mask, multiples=[1, max_triplets, 2])
      permutations_mask = neighbours_pairs_combinations < permutations_mask
      permutations_mask = tf.reduce_all(permutations_mask, axis=-1)

      neighbours_pairs_combinations = tf.gather_nd(neighbours_pairs_combinations, tf.where(permutations_mask))
      neighbours_pairs_combinations += tf.expand_dims(tf.repeat(index_shifts, (neighbours_count*(neighbours_count-1)//2)), axis=-1)

      inside_cutoff_indices = tf.where(cutoff_mask)
      flattened_triplets_indices = tf.gather(inside_cutoff_indices, neighbours_pairs_combinations)

      j_indices, k_indices = tf.unstack(flattened_triplets_indices, axis=1)
      flipped_k_indices = tf.reverse(k_indices, axis=[1])

      messy_indices = tf.concat([j_indices, flipped_k_indices], axis=-1)
      jk_indices = tf.slice(messy_indices, [0, 2], [tf.shape(messy_indices)[0], 2])

      batch_indices = tf.slice(messy_indices, [0, 0], [tf.shape(messy_indices)[0], 1])

      jk_indices = tf.concat([batch_indices, jk_indices], axis=-1)

      jk_indices = tf.expand_dims(jk_indices, axis=1)
      flattened_triplets_indices = tf.concat([flattened_triplets_indices, jk_indices], axis=1)

      return flattened_triplets_indices, neighbours_count

   def call(self, distance_matrices, num_species_batch):

      # 1. Set some environment variables
      batch_size = tf.shape(num_species_batch)[0]
      num_atoms = tf.shape(num_species_batch)[1]

      # 2. Get the distances' triplets
      cutoff_mask = (distance_matrices < self.cutoff) & (distance_matrices != 0.)
      flattened_triplets_indices, neighbours_count = self.computeDistancesTripletsIndices(cutoff_mask)
      interesting_triplets_distances = tf.gather_nd(distance_matrices, flattened_triplets_indices)   # (batch_size*num_atoms*interesting_triplets, 3)

      # 3. Get the species' triplets
      flattened_species_indices = self.computeSpeciesTripletsIndices(flattened_triplets_indices)
      interesting_triplets_species = tf.gather_nd(num_species_batch, flattened_species_indices)      # (batch_size*num_atoms*interesting_triplets, 3)

      # 4. Compute the GA addends
      interesting_geometries = self.carnotTheorem(interesting_triplets_distances)
      interesting_GA_addends = self.computeGATerms(interesting_geometries, self.quadruplets_params_a)
      interesting_GA_addends = tf.transpose(interesting_GA_addends)                                            # (batch_size*num_atoms*interesting_triplets, num_quadruplets_params_a)

      # 5. Reconstruct the (minimal) shape and sum for the output
      interesting_species_jk = tf.slice(interesting_triplets_species, begin=[0, 1], size=[-1, 2])                # (batch_size*num_atoms*interesting_triplets, 2)
      interesting_species_jk = tf.expand_dims(interesting_species_jk, axis=1)
      interesting_species_jk = tf.tile(interesting_species_jk, multiples=[1, len(self.pairs_vocabulary), 1])     # (batch_size*num_atoms*interesting_triplets, num_pairs_voc, 2)
      pairs_vocabulary_tensor = tf.constant(self.pairs_vocabulary, dtype=interesting_species_jk.dtype)
      interesting_species_one_hot = tf.reduce_all(
                                          interesting_species_jk==pairs_vocabulary_tensor, axis=-1
                                          ) | tf.reduce_all(
                                                   interesting_species_jk==tf.reverse(pairs_vocabulary_tensor, [-1]), axis=-1
                                                   )                                                             # (batch_size*num_atoms*interesting_triplets, num_pairs_voc)
      indices_for_sum = tf.where(interesting_species_one_hot)
      indices_for_sum = tf.slice(indices_for_sum, [0, 1], [-1, 1])
      GA_batch = tf.zeros((batch_size*num_atoms, len(self.pairs_vocabulary), self.num_quadruplets_params_a))
      batch_indices_for_sum = tf.range(batch_size*num_atoms, dtype=indices_for_sum.dtype)
      triplets_count = neighbours_count * (neighbours_count-1) // 2
      batch_indices_for_sum = tf.repeat(batch_indices_for_sum, triplets_count, axis=0)
      indices_for_sum = tf.concat([tf.expand_dims(batch_indices_for_sum, axis=1), indices_for_sum], axis=1)
      GA_batch = tf.tensor_scatter_nd_add(GA_batch, indices_for_sum, interesting_GA_addends)
      GA_batch = tf.reshape(GA_batch, [batch_size, num_atoms, -1])

      return GA_batch

#
##
### NON-LEARNABLE LAYERS
##
#

class PBCCoordToDist(tf.keras.layers.Layer):

   def __init__(self):
      super(PBCCoordToDist, self).__init__()

   def call(self, coordinates_batch, box_sizes):

      t1 = tf.expand_dims(coordinates_batch, axis=1)
      t2 = tf.expand_dims(coordinates_batch, axis=2)

      per_dim_absolute_differences = tf.abs( t1-t2 )
      per_dim_actual_differences = tf.math.minimum(per_dim_absolute_differences, box_sizes-per_dim_absolute_differences)

      distances = tf.sqrt( tf.reduce_sum(per_dim_actual_differences**2, axis=-1) )

      return distances

class CoordToDist(tf.keras.layers.Layer):

   def __init__(self):
      super(CoordToDist, self).__init__()

   @staticmethod
   def call(coordinates_batch):

      t1 = tf.expand_dims(coordinates_batch, axis=1)
      t2 = tf.expand_dims(coordinates_batch, axis=2)

      per_dim_absolute_differences = tf.abs( t1-t2 )

      distances = tf.sqrt( tf.reduce_sum(per_dim_absolute_differences**2, axis=-1) )
      
      return distances

class CELU(tf.keras.layers.Layer):
   
   def __init__(self, alpha):
      super(CELU, self).__init__()
      self.alpha = alpha

   def call(self, x):
      zero = tf.constant(0., dtype=x.dtype)
      safe_x = tf.where(x>zero, zero, x)      # due to a TensorFlow bug
      return tf.where(x>0., x, self.alpha * (tf.math.exp(safe_x/self.alpha) - 1))

#
##
### MODELS
##
#

class Anakin(tf.keras.Model):

   def __init__(self,
                  dynamic_mode = False,
                  weights = None,
               ):
      super(Anakin, self).__init__(name="anakin")
      """Space units = Angstroms. Energy units = Hartree."""

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

      # Self energies
      self.self_energies = tf.constant(self_energies_list)

      # AEV computers
      self.radial_aev_computer = RadialAEVComputer(vocabulary, radial_AEV_params)
      self.angular_aev_computer = AngularAEVComputer(vocabulary, angular_AEV_params)

      # torchani-2x MLPs architectures (matched with the vocabulary)
      radial_aev_lenght = len(vocabulary) * len(radial_AEV_params['EtaR'])*len(radial_AEV_params['ShfR'])
      angular_aev_lenght = len(vocabulary)*(len(vocabulary)+1)/2 * len(angular_AEV_params['Zeta'])*len(angular_AEV_params['ShfZ'])*len(angular_AEV_params['EtaA'])*len(angular_AEV_params['ShfA'])
      self.aev_lenght = int(radial_aev_lenght + angular_aev_lenght)
      self.H_network = self.getMLP([256, 192, 160])
      self.C_network = self.getMLP([224, 192, 160])
      self.N_network = self.getMLP([192, 160, 128])
      self.O_network = self.getMLP([192, 160, 128])
      self.F_network = self.getMLP([160, 128, 96])
      self.S_network = self.getMLP([160, 128, 96])
      self.Cl_network = self.getMLP([160, 128, 96])

      if dynamic_mode:     # dynamic mode = PBC in a box; atomic numbers as input; no dummy atoms (=> custom __call__)
         self.distances_computer = PBCCoordToDist()
         self.species_translator = tf.keras.layers.StringLookup(vocabulary=vocabulary, output_mode='int', mask_token='', num_oov_indices=0)
         self.call = self.computeEnergyForDynamics
      else:
         self.distances_computer = CoordToDist()
         self.species_translator = tf.keras.layers.StringLookup(vocabulary=vocabulary, output_mode='int', mask_token='', num_oov_indices=0)
         self.call = self.computeEnergy

      # Load the weights if provided
      if weights is not None:
         self.loadWeights(weights, dynamic_mode)

   def getMLP(self, neurons_per_hidden_layer):

      inputs = Input(shape=(self.aev_lenght))

      x = Dense(neurons_per_hidden_layer[0], kernel_constraint=max_norm(3.), kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
      x = CELU(0.1)(x)
      for neurons in neurons_per_hidden_layer[1:]:
         x = Dense(neurons, kernel_constraint=max_norm(3.), kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
         x = CELU(0.1)(x)
      
      outputs = Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)

      return keras.Model(inputs=inputs, outputs=outputs)
   
   def loadWeights(self, model_path, dynamic_mode):

      if dynamic_mode:
         self.build( [(None, None, 3), (None, None), (None, 3)] )
      else:
         self.build( [(None, None, 3), (None, None)] )

      model_for_weights = tf.keras.models.load_model(model_path)
      model_for_weights.trainable = True     # in case these weights come from a fine-tuning
      self.set_weights(model_for_weights.get_weights())

   def computeMLPEnergies(self, masked_aevs, masked_num_species):
      """Not optimized, but also not a problem for state-of-the-art parallelization hardware."""

      H_energies = tf.reshape( self.H_network(masked_aevs), [-1] )
      C_energies = tf.reshape( self.C_network(masked_aevs), [-1] )
      N_energies = tf.reshape( self.N_network(masked_aevs), [-1] )
      O_energies = tf.reshape( self.O_network(masked_aevs), [-1] )
      F_energies = tf.reshape( self.F_network(masked_aevs), [-1] )
      S_energies = tf.reshape( self.S_network(masked_aevs), [-1] )
      Cl_energies = tf.reshape( self.Cl_network(masked_aevs), [-1] )

      stacked_energies = tf.stack([H_energies, C_energies, N_energies, O_energies, F_energies, S_energies, Cl_energies], axis=-1)
      indices = masked_num_species-1
      indices = tf.stack([tf.range(tf.shape(indices)[0]), indices], axis=-1)
      masked_mlp_energies = tf.gather_nd(stacked_energies, indices)

      return masked_mlp_energies

   def computeEnergy(self, inputs, training=None, mask=None):
      """Compute the molecular energies from the input coordinates and species."""

      coordinates_batch, string_species_batch = inputs
      batch_size = tf.shape(coordinates_batch)[0]
      num_atoms = tf.shape(coordinates_batch)[1]

      # 1. Preprocess the inputs
      distance_matrices = self.distances_computer(coordinates_batch)
      num_species_batch = tf.cast(self.species_translator(string_species_batch), dtype=self.dtype)
      dummy_mask = num_species_batch != 0.     # (batch_size, num_atoms)

      # 2. Compute and concatenate the AEVs
      rad_aevs_batch = self.radial_aev_computer(distance_matrices, num_species_batch)
      ang_aevs_batch = self.angular_aev_computer(distance_matrices, num_species_batch)
      aevs_batch = tf.concat([rad_aevs_batch, ang_aevs_batch], axis=-1)

      # 3. Compute the per-atom self and MLP energies
      masked_num_species = tf.cast(tf.boolean_mask(num_species_batch, dummy_mask), dtype=tf.int32)    # (num_nondummy_atoms,)
      masked_aevs = tf.boolean_mask(aevs_batch, dummy_mask)
      masked_self_energies = tf.gather(self.self_energies, masked_num_species-1)          # -1 to have H 0-indexed
      masked_mlp_energies = self.computeMLPEnergies(masked_aevs, masked_num_species)      # (num_nondummy_atoms,)

      # 4. Reconstruct the shape and sum to obtain the molecular energies
      masked_complete_atomic_energies = masked_self_energies + masked_mlp_energies
      complete_atomic_energies = tf.zeros([batch_size, num_atoms], dtype=self.dtype)
      complete_atomic_energies = tf.tensor_scatter_nd_add(complete_atomic_energies, tf.where(dummy_mask), masked_complete_atomic_energies)
      molecular_energies = tf.reduce_sum(complete_atomic_energies, -1)

      return molecular_energies
   
   def computeEnergyForDynamics(self, inputs, training=None, mask=None):

      coordinates_batch, string_species_batch, box_sizes = inputs    # it receives string species also in dynamic mode
      batch_size = tf.shape(coordinates_batch)[0]
      num_atoms = tf.shape(coordinates_batch)[1]

      # 1. Preprocess the inputs
      distance_matrices = self.distances_computer(coordinates_batch, box_sizes)
      num_species_batch = tf.cast(self.species_translator(string_species_batch), dtype=self.dtype)

      # 2. Compute and concatenate the AEVs
      rad_aevs_batch = self.radial_aev_computer(distance_matrices, num_species_batch)
      ang_aevs_batch = self.angular_aev_computer(distance_matrices, num_species_batch)
      aevs_batch = tf.concat([rad_aevs_batch, ang_aevs_batch], axis=-1)

      # 3. Compute the per-atom self and MLP energies
      flattened_aevs = tf.reshape(aevs_batch, [batch_size*num_atoms, -1])
      flattened_num_species = tf.cast( tf.reshape(num_species_batch, [-1]), dtype=tf.int32 )
      flattened_self_energies = tf.gather(self.self_energies, flattened_num_species-1)          # -1 to have H 0-indexed
      flattened_mlp_energies = self.computeMLPEnergies(flattened_aevs, flattened_num_species)   # (num_nondummy_atoms,)

      # 4. Reconstruct the shape and sum to obtain the molecular energies
      flattened_complete_atomic_energies = flattened_self_energies + flattened_mlp_energies
      complete_atomic_energies = tf.reshape(flattened_complete_atomic_energies, [batch_size, num_atoms])
      molecular_energies = tf.reduce_sum(complete_atomic_energies)

      return molecular_energies

   def computeEnergyAndForces(self, inputs):
      """Computation is done in batches."""

      coords = inputs[0]

      with tf.GradientTape(watch_accessed_variables=False) as force_tape:
         force_tape.watch(coords)
         energy = self.call(inputs)    # energies computed for free
      forces = -force_tape.gradient(energy, coords)

      return energy, forces
