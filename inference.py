from architectures import net_utils
import tensorflow as tf

model = net_utils.getModel(model_name="obiwan", weights="results/models/obiwan_ani1Uani2_FH_VL_2.404")

coordinates = tf.constant([[0.0, 0.0, 1.0], [0.0, 1.2, 1.0], [0.0, 2.4, 1.0]])
species = tf.constant(['O', 'C', 'O'])

# We have to add an outer dimension because OBI expects a 'series' of molecules
batched_coordinates = tf.expand_dims(coordinates, axis=0)
batched_species = tf.expand_dims(species, axis=0)

# Energy are in Hartrees
batched_energies, batched_forces = model.computeEnergyAndForces((batched_coordinates, batched_species))
print(batched_energies)
print(batched_forces)
