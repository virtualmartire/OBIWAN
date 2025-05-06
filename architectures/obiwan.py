import tensorflow as tf
import math
import numpy as np

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Add
from keras.constraints import max_norm

#
##
### AEVComputers
##
#

def f_C(cutoff, distance):
   """The ANAKIN-ME continuous cutoff function. Outliers will be mapped to 0 in another way."""
   return 0.5 * tf.math.cos(math.pi * distance / cutoff) + 0.5

class DeepRadAEVComputer(tf.keras.layers.Layer):

   def __init__(self,
                  radial_cutoff,
               ):
      super(DeepRadAEVComputer, self).__init__(name="deep_rad_aev_computer")

      self.cutoff = tf.cast(radial_cutoff, self.dtype)

      self.radMLP = self.getMLP()
      self.mlp_output_dim = self.radMLP.layers[-1].units

      self.f_C = lambda distance: f_C(self.cutoff, distance)

   @staticmethod
   def getMLP():

      inputs = Input(3)

      x_res = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="rad_aev_dense_0")(inputs)
      x = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="rad_aev_dense_1")(x_res)
      x_block_1 = Add()([x, x_res])

      x = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="rad_aev_dense_2")(x_block_1)
      x = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="rad_aev_dense_3")(x)
      x_block_2 = Add()([x, x_block_1])

      x = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="rad_aev_dense_4")(x_block_2)
      x = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="rad_aev_dense_5")(x)
      x_block_3 = Add()([x, x_block_2])

      outputs = Dense(128, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="rad_aev_dense_6")(x_block_3)

      return Model(inputs, outputs)

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

   def computeGRTerms(self, interesting_distances, interesting_species_pairs):
      """Our deep AEV formulation for the radial components."""

      R_ij = interesting_distances
      z_i, z_j = tf.unstack(interesting_species_pairs, axis=-1)

      # 1. Build the input, normalizing the chemical components
      geometric_input = tf.expand_dims(R_ij, -1)
      chemical_input = tf.stack([z_i+z_j, z_i*z_j], axis=-1)
      chemical_input = chemical_input / ( tf.linalg.norm(chemical_input, axis=-1, keepdims=True)+tf.keras.backend.epsilon() )
      mlp_input = tf.concat([geometric_input, chemical_input], axis=-1)

      # 2. Forward pass
      mlp_output = self.radMLP(mlp_input)

      # 3. Cutoff smoothing
      cutoff_smoothing = self.f_C(R_ij)
      cutoff_smoothing = tf.expand_dims(cutoff_smoothing, -1)

      return mlp_output*cutoff_smoothing

   @staticmethod
   def tileSpeciesMatrix(atomic_numbers_batch):

      num_atoms = tf.shape(atomic_numbers_batch)[1]

      i_atomic_numbers_batch = tf.expand_dims(atomic_numbers_batch, axis=-1)
      i_matrix_atomic_numbers_batch = tf.tile(i_atomic_numbers_batch, multiples=[1, 1, num_atoms])
      j_atomic_numbers_batch = tf.expand_dims(atomic_numbers_batch, axis=-2)
      j_matrix_atomic_numbers_batch = tf.tile(j_atomic_numbers_batch, multiples=[1, num_atoms, 1])

      return tf.stack([i_matrix_atomic_numbers_batch, j_matrix_atomic_numbers_batch], axis=-1)

   def call(self, distance_matrices_batch, atomic_numbers_batch):

      # 1. Compute and apply the cutoff mask
      cutoff_mask = (distance_matrices_batch < self.cutoff) & (distance_matrices_batch != 0.)      # != 0. to exclude dummy atoms, that are all in the same place
      interesting_distances = tf.boolean_mask(distance_matrices_batch, cutoff_mask)
      matrix_atnum_batch = self.tileSpeciesMatrix(atomic_numbers_batch)
      interesting_atnum_pairs = tf.boolean_mask(matrix_atnum_batch, cutoff_mask)

      # 2. Compute the GR addends
      interesting_GR_addends = self.computeGRTerms(interesting_distances, interesting_atnum_pairs)       # (batch_size*num_atoms*interesting_pairs, mlp_output_dim)

      # 3. Reconstruct the shape and sum
      GR_addends = self.createFakeRaggedTensor(interesting_GR_addends, cutoff_mask)                      # (batch_size, num_atoms, fake_ragged_dim_length, mlp_output_dim)
      GR_batch = tf.reduce_sum(GR_addends, axis=-2)                                                      # (batch_size, num_atoms, mlp_output_dim)

      # 4. Normalize
      GR_batch = GR_batch / ( tf.linalg.norm(GR_batch, axis=-1, keepdims=True)+tf.keras.backend.epsilon() )

      return GR_batch


class DeepAngAEVComputer(tf.keras.layers.Layer):

   def __init__(self,
                  angular_cutoff,
               ):
      super(DeepAngAEVComputer, self).__init__(name="deep_ang_aev_computer")

      self.cutoff = tf.cast(angular_cutoff, self.dtype)

      self.angMLP = self.getMLP()
      self.mlp_output_dim = self.angMLP.layers[-1].units

      self.f_C = lambda distance: f_C(self.cutoff, distance)

   @staticmethod
   def getMLP():

      inputs = Input(9)

      x_res = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="ang_aev_dense_0")(inputs)
      x = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="ang_aev_dense_1")(x_res)
      x_block_1 = Add()([x, x_res])

      x = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="ang_aev_dense_2")(x_block_1)
      x = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="ang_aev_dense_3")(x)
      x = Dense(64, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="ang_aev_dense_4")(x)
      x_block_2 = Add()([x, x_block_1])

      x_block_3 = Dense(128, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="ang_aev_dense_5")(x_block_2)

      outputs = Dense(256, activation='tanh', kernel_initializer='he_normal', bias_initializer='zeros', name="ang_aev_dense_6")(x_block_3)

      return Model(inputs, outputs)

   @staticmethod
   def computeCosAngles(R_ij, R_ik, R_jk):
      """Implementation of the Carnot formula in order to compute angles from the input distances."""
      cos_theta_i = (R_ij**2 + R_ik**2 - R_jk**2) / tf.clip_by_value(2*R_ij*R_ik, clip_value_min=1e-10, clip_value_max=np.inf)
      cos_theta_j = (R_ij**2 + R_jk**2 - R_ik**2) / tf.clip_by_value(2*R_ij*R_jk, clip_value_min=1e-10, clip_value_max=np.inf)
      cos_theta_k = (R_ik**2 + R_jk**2 - R_ij**2) / tf.clip_by_value(2*R_ik*R_jk, clip_value_min=1e-10, clip_value_max=np.inf)
      return cos_theta_i, cos_theta_j, cos_theta_k

   def computeGATerms(self, triplets_distances, triplets_species):
      """Our deep AEV formulation for the angular components."""
      
      R_ij, R_ik, R_jk = tf.unstack(triplets_distances, axis=-1)
      z_i, z_j, z_k = tf.unstack(triplets_species, axis=-1)
      cos_theta_i, cos_theta_j, cos_theta_k = self.computeCosAngles(R_ij, R_ik, R_jk)

      # 1. Build the input computing the coefficients of the complex polynomial in question utilising the Viète-Girard formulas
      geometric_input = tf.stack(
                                 [R_ij+R_ik+R_jk,
                                 R_ij*R_ik + R_ij*R_jk + R_ik*R_jk,
                                 R_ij*R_ik*R_jk], axis=-1)
      chemical_input = tf.stack(
                                 [z_i+z_j+z_k,
                                 cos_theta_i+cos_theta_j+cos_theta_k,
                                 z_i*(z_j+z_k) + z_j*z_k - cos_theta_i*(cos_theta_j+cos_theta_k) - cos_theta_j*cos_theta_k,
                                 z_i*(cos_theta_j+cos_theta_k) + cos_theta_i*(z_j+z_k) + z_j*cos_theta_k + cos_theta_j*z_k,
                                 z_i*(z_j*z_k - cos_theta_j*cos_theta_k) - cos_theta_i*(z_j*cos_theta_k + cos_theta_j*z_k),
                                 z_i*(z_j*cos_theta_k + cos_theta_j*z_k) + cos_theta_i*(z_j*z_k - cos_theta_j*cos_theta_k)], axis=-1)
      
      # 2. Normalize (separately in order to manage the different orders of magnitude) and concatenate
      geometric_input = geometric_input / ( tf.linalg.norm(geometric_input, axis=-1, keepdims=True)+tf.keras.backend.epsilon() )
      chemical_input = chemical_input / ( tf.linalg.norm(chemical_input, axis=-1, keepdims=True)+tf.keras.backend.epsilon() )
      mlp_input = tf.concat([geometric_input, chemical_input], axis=-1)

      # 3. Forward pass
      mlp_output = self.angMLP(mlp_input)                               # (num_triplets, mlp_output_dim)

      # 4. Cutoff smoothing
      cutoff_smoothing = self.f_C(R_ij) * self.f_C(R_ik)                # (num_triplets,)
      cutoff_smoothing = tf.expand_dims(cutoff_smoothing, -1)           # needed to guide the broadcasting

      return mlp_output*cutoff_smoothing
   
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

      # 4. Compute the G addends
      interesting_GA_addends = self.computeGATerms(interesting_triplets_distances, interesting_triplets_species)  # (batch_size*num_atoms*interesting_triplets, mlp_output_dim)

      # 5. Reconstruct the shape and sum
      GA_batch = tf.zeros((batch_size, num_atoms, self.mlp_output_dim))
      indices_for_sum = tf.where( tf.reduce_all(GA_batch == [0.]*self.mlp_output_dim, axis=-1) )
      triplets_count = neighbours_count * (neighbours_count-1) // 2
      indices_for_sum = tf.repeat(indices_for_sum, triplets_count, axis=0)
      GA_batch = tf.tensor_scatter_nd_add(GA_batch, indices_for_sum, interesting_GA_addends)

      # 6. Normalize
      GA_batch = GA_batch / ( tf.linalg.norm(GA_batch, axis=-1, keepdims=True)+tf.keras.backend.epsilon() )

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

class Obiwan(tf.keras.Model):

   def __init__(self,
                  radial_cutoff = 5.2,
                  angular_cutoff = 3.5,
                  dynamic_mode = False,
                  weights = None,
               ):
      super(Obiwan, self).__init__(name="obiwan")
      """Space units = Angstroms. Energy units = Hartree."""

      periodic_table = [['H', -0.49979], 'He', 'Li', 'Be', 'B', ['C', -37.78942], ['N', -54.52998], ['O', -75.00475],
                        ['F', -99.66838], 'Ne', 'Na', 'Mg', 'Al', 'Si', ['P', -340.83185], ['S', -397.6754], ['Cl', -459.70697],
                        'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
                        ['Br', -2574.11672], 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                        'Sn', 'Sb', 'Te', ['I', -297.76229], 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                        'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                        'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                        'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo'] 
                        # we provide the entire periodic table so that each position is equal to the atomic number;
                        # for Br and I we used DFT approximations.

      # CCSD(T) self-energies
      ccsdt_list = [element[1] if isinstance(element, list) else 0. for element in periodic_table]
      self.ccsdt_self_energies = tf.constant(ccsdt_list, dtype=self.dtype)

      # Deep AEV computers
      self.deep_rad_aev_computer = DeepRadAEVComputer(radial_cutoff)
      self.deep_ang_aev_computer = DeepAngAEVComputer(angular_cutoff)

      # Single MLP
      aev_length = self.deep_rad_aev_computer.mlp_output_dim+self.deep_ang_aev_computer.mlp_output_dim
      self.single_mlp = self.getSingleMLP(aev_length)

      if dynamic_mode:     # dynamic mode = PBC in a box; atomic numbers as input; no dummy atoms (=> custom __call__)
         self.distances_computer = PBCCoordToDist()
         self.call = self.computeEnergyForDynamics
      else:
         self.distances_computer = CoordToDist()
         vocabulary = [element[0] if isinstance(element, list) else element for element in periodic_table]
         self.species_translator = tf.keras.layers.StringLookup(vocabulary=vocabulary, output_mode='int', mask_token='', num_oov_indices=0)
         self.call = self.computeEnergy

      # Load the weights if provided
      if weights is not None:
         self.loadWeights(weights, dynamic_mode)

   @staticmethod
   def getSingleMLP(aev_length):

      inputs = Input(aev_length)

      x = Dense(1024, kernel_constraint=max_norm(3.), kernel_initializer='he_normal', bias_initializer='zeros', name="single_mlp_dense_0")(inputs)
      x = CELU(0.1)(x)
      x = Dense(768, kernel_constraint=max_norm(3.), kernel_initializer='he_normal', bias_initializer='zeros', name="single_mlp_dense_1")(x)
      x = CELU(0.1)(x)
      x = Dense(512, kernel_constraint=max_norm(3.), kernel_initializer='he_normal', bias_initializer='zeros', name="single_mlp_dense_2")(x)
      x = CELU(0.1)(x)
      x = Dense(256, kernel_constraint=max_norm(3.), kernel_initializer='he_normal', bias_initializer='zeros', name="single_mlp_dense_3")(x)
      x = CELU(0.1)(x)

      outputs = Dense(1, kernel_initializer='he_normal', bias_initializer='zeros', name="single_mlp_dense_4")(x)

      return Model(inputs, outputs, name="single_mlp")

   def loadWeights(self, model_path, dynamic_mode = False):

      if dynamic_mode:
         self.build( [(None, None, 3), (None, None), (None, 3)] )
      else:
         self.build( [(None, None, 3), (None, None)] )

      model_for_weights = tf.keras.models.load_model(model_path)
      model_for_weights.trainable = True     # in case these weights come from a fine-tuning
      self.set_weights(model_for_weights.get_weights())

   def freezeFirstLayers(self):

      self.deep_rad_aev_computer.radMLP.get_layer('rad_aev_dense_0').trainable = False
      self.deep_rad_aev_computer.radMLP.get_layer('rad_aev_dense_1').trainable = False
      self.deep_rad_aev_computer.radMLP.get_layer('rad_aev_dense_2').trainable = False
      self.deep_rad_aev_computer.radMLP.get_layer('rad_aev_dense_3').trainable = False
      self.deep_rad_aev_computer.radMLP.get_layer('rad_aev_dense_4').trainable = False

      self.deep_ang_aev_computer.angMLP.get_layer('ang_aev_dense_0').trainable = False
      self.deep_ang_aev_computer.angMLP.get_layer('ang_aev_dense_1').trainable = False
      self.deep_ang_aev_computer.angMLP.get_layer('ang_aev_dense_2').trainable = False
      self.deep_ang_aev_computer.angMLP.get_layer('ang_aev_dense_3').trainable = False
      self.deep_ang_aev_computer.angMLP.get_layer('ang_aev_dense_4').trainable = False

      self.single_mlp.get_layer('single_mlp_dense_0').trainable = False
      self.single_mlp.get_layer('single_mlp_dense_1').trainable = False

   def computeEnergy(self, inputs, training=None, mask=None):
      """IMPORTANT: dummy atoms must be at 'infinity' and with species '' to ensure
      the correct action of the masking mechanisms.
      
      Outputs are in Hartree."""

      coordinates_batch, string_species_batch = inputs
      batch_size = tf.shape(coordinates_batch)[0]
      num_atoms = tf.shape(coordinates_batch)[1]

      # 1. Preprocess the inputs
      distance_matrices = self.distances_computer(coordinates_batch)
      atomic_numbers_batch = tf.cast(self.species_translator(string_species_batch), dtype=self.dtype)
      dummy_mask = atomic_numbers_batch != 0.     # (batch_size, num_atoms)

      # 2. Compute and concatenate the AEVs
      rad_aevs_batch = self.deep_rad_aev_computer(distance_matrices, atomic_numbers_batch)
      ang_aevs_batch = self.deep_ang_aev_computer(distance_matrices, atomic_numbers_batch)
      aevs_batch = tf.concat([rad_aevs_batch, ang_aevs_batch], axis=-1)

      # 3. Compute the per-atom self and mlp energies
      masked_aevs = tf.boolean_mask(aevs_batch, dummy_mask)
      masked_atomic_numbers = tf.cast(tf.boolean_mask(atomic_numbers_batch, dummy_mask), dtype=tf.int32)
      masked_self_energies = tf.expand_dims(
                                 tf.gather(self.ccsdt_self_energies, masked_atomic_numbers-1),
                                 axis=-1
                              )
      masked_mlp_energies = self.single_mlp(masked_aevs)

      # 4. Compute the sums to obtain molecular energies
      masked_complete_atomic_energies = masked_self_energies + masked_mlp_energies
      complete_atomic_energies = tf.zeros([batch_size, num_atoms, 1], dtype=self.dtype)      # at this point we have to reconstruct the shape
      complete_atomic_energies = tf.tensor_scatter_nd_add(complete_atomic_energies, tf.where(dummy_mask), masked_complete_atomic_energies)
      molecular_energies = tf.reduce_sum(tf.squeeze(complete_atomic_energies), -1)

      return molecular_energies
   
   def computeEnergyForDynamics(self, inputs, training=None, mask=None):

      coordinates_batch, atomic_numbers_batch, box_sizes = inputs
      batch_size = tf.shape(coordinates_batch)[0]
      num_atoms = tf.shape(coordinates_batch)[1]

      # 1. Preprocess the inputs
      distance_matrices = self.distances_computer(coordinates_batch, box_sizes)
      atomic_numbers_batch = tf.cast(atomic_numbers_batch, dtype=self.dtype)

      # 2. Compute and concatenate the AEVs
      rad_aevs_batch = self.deep_rad_aev_computer(distance_matrices, atomic_numbers_batch)
      ang_aevs_batch = self.deep_ang_aev_computer(distance_matrices, atomic_numbers_batch)
      aevs_batch = tf.concat([rad_aevs_batch, ang_aevs_batch], axis=-1)

      # 3. Compute the per-atom self and mlp energies
      flattened_aevs = tf.reshape(aevs_batch, [batch_size*num_atoms, -1])
      flattened_atomic_numbers = tf.reshape(atomic_numbers_batch, [-1])
      flattened_atomic_numbers = tf.cast(flattened_atomic_numbers, dtype=tf.int32)
      flattened_mlp_energies = self.single_mlp(flattened_aevs)
      flattened_self_energies = tf.expand_dims(
                                 tf.gather(self.ccsdt_self_energies, flattened_atomic_numbers-1),
                                 axis=-1
                              )

      # 4. Compute the sums to obtain molecular energies
      flattened_complete_atomic_energies = flattened_self_energies + flattened_mlp_energies
      complete_atomic_energies = tf.reshape(flattened_complete_atomic_energies, [batch_size, num_atoms, 1])
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