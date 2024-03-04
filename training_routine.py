import json
import tensorflow as tf
import time
import argparse
import os

import architectures.net_utils as net_utils
import data.datasetting as datasetting

#
##
### HYPERPARAMETERS
##
#

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--batch_size_per_worker', type=int, default=None)
parser.add_argument('--resume_training', action='store_true')     # store_true action to set a boolean flag to True if the argument is present, and to False otherwise
parser.add_argument('--datasets', type=str, nargs='+', default=None)
parser.add_argument('--with_forces', type=str2bool, nargs='+', default=None)
parser.add_argument('--remove_toxic_molecules', type=str2bool, nargs='+', default=None)

parser.add_argument('--custom_resume', action='store_true')
parser.add_argument('--last_epoch', type=int, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--new_learning_rate', type=float, default=None)
parser.add_argument('--freeze_first_layers', action='store_true')

args = parser.parse_args()

# Resume training
run_name = args.run_name
resume_training = args.resume_training
custom_resume = args.custom_resume
if resume_training is True:
  if custom_resume is False:
    # read the last epoch index from the log file
    with open(f"results/logs/{run_name}.json", "r") as f:
      history = json.load(f)[run_name]
      last_epoch = len(history[list(history.keys())[0]])
    # resume from the last checkpoint
    model_path = f"results/models/last_model_{run_name}"
  else:       # set the resume parameters manually
    last_epoch = args.last_epoch
    new_learning_rate = args.new_learning_rate
    model_path = args.model_path
else:
  last_epoch = 0
freeze_first_layers = args.freeze_first_layers

# Batch size (programmatically built)
num_gpus = len(tf.config.list_physical_devices('GPU'))
batch_size_per_worker = args.batch_size_per_worker
if num_gpus > 0:
  batch_size = batch_size_per_worker * num_gpus
else:
  batch_size = batch_size_per_worker

# Adam
initial_learning_rate_bs64 = 2e-5
initial_learning_rate = initial_learning_rate_bs64 * tf.math.sqrt(batch_size / 64)
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.00001

# Data
data_dtype_name = "float32"
data_dtype = datasetting.getTfDtype(data_dtype_name)

# Model
model_name = args.model_name

#
##
### MODEL
##
#

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():

  model = net_utils.getModel(model_name)

  if resume_training:
    # Transfer the weights
    model.loadWeights(model_path)
    if freeze_first_layers:
      model.freezeFirstLayers()
    # Transfer the optimizer
    model_for_weights = tf.keras.models.load_model(model_path)
    optimizer = model_for_weights.optimizer
    if custom_resume:
      optimizer.lr = new_learning_rate

  else:
    model.build( [(None, None, 3), (None, None)] )
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon, weight_decay=weight_decay)
  
  model.compile(optimizer=optimizer)      # we compile the optimizer with the model to be able to save its state

model.summary()

#
##
### INPUT PIPELINE (80% train, 20% validation)
##
#

datasets_names_list = args.datasets
with_forces_list = args.with_forces
remove_toxic_molecules_list = args.remove_toxic_molecules

training_sets_list = []
validation_sets_list = []
for ds_name, with_forces, remove_toxic_molecules in zip(datasets_names_list, with_forces_list, remove_toxic_molecules_list):

  # Load the datasets (already shuffled)
  training_set, validation_set = datasetting.getDatasets(ds_name, data_dtype_name)

  # Remove the toxic molecules from the training set in case
  if remove_toxic_molecules is True:
    training_set = training_set.padded_batch(
                                              batch_size=1,
                                              padding_values=((tf.constant(1e20, dtype=tf.float32), ""),
                                                              (None, tf.constant(0., dtype=tf.float32))),
                                              drop_remainder=True
                                            )
    training_set = training_set.filter(lambda x, y: net_utils.is_not_toxic(model, x, y))    # model-dependent
    training_set = training_set.unbatch()
    training_set_path = f"data/datasets/{ds_name}/cache/{data_dtype_name}/training_healed_{model_name}/"
    os.makedirs(training_set_path, exist_ok=True)
    training_set = training_set.cache(training_set_path)

  # Remove forces in case (but add dummy ones--because, in order to be fused, datasets must have the same shape)
  if with_forces is False:
    training_set = training_set.map(lambda x, y: ( x, (y[0], tf.zeros_like(x[0])) ))
    validation_set = validation_set.map(lambda x, y: ( x, (y[0], tf.zeros_like(x[0])) ))

  # Batch
  training_set = training_set.padded_batch(batch_size, padding_values=((tf.constant(1e20, dtype=tf.float32), ""),
                                                                        (None, tf.constant(0., dtype=tf.float32))),
                                                        drop_remainder=True)
  validation_set = validation_set.padded_batch(batch_size, padding_values=((tf.constant(1e20, dtype=tf.float32), ""),
                                                                            (None, tf.constant(0., dtype=tf.float32))),
                                                            drop_remainder=True)

  # Append
  training_sets_list.append(training_set)
  validation_sets_list.append(validation_set)

# Fused the datasets
choice_training_set = tf.data.Dataset.range(len(training_sets_list)).repeat()
training_set = tf.data.Dataset.choose_from_datasets(training_sets_list, choice_training_set, stop_on_empty_dataset=False)
choice_validation_set = tf.data.Dataset.range(len(validation_sets_list)).repeat()
validation_set = tf.data.Dataset.choose_from_datasets(validation_sets_list, choice_validation_set, stop_on_empty_dataset=False)

#
##
### TRAINING UTILITIES
##
#

# Metrics
energyRMSE_tracker = tf.keras.metrics.Mean(name="energyRMSE")              # they are a mean of nothing because we will pass scalars to them
energyRMSE_atom_tracker = tf.keras.metrics.Mean(name="energyRMSE_atom")
forcesRMSE_tracker = tf.keras.metrics.Mean(name="forcesRMSE")
forcesRMSE_atom_tracker = tf.keras.metrics.Mean(name="forcesRMSE_atom")
val_energyRMSE_tracker = tf.keras.metrics.Mean(name="val_energyRMSE")
val_energyRMSE_atom_tracker = tf.keras.metrics.Mean(name="val_energyRMSE_atom")
val_forcesRMSE_tracker = tf.keras.metrics.Mean(name="val_forcesRMSE")
val_forcesRMSE_atom_tracker = tf.keras.metrics.Mean(name="val_forcesRMSE_atom")
metrics_list = [energyRMSE_tracker, energyRMSE_atom_tracker, forcesRMSE_tracker, forcesRMSE_atom_tracker, val_energyRMSE_tracker, val_energyRMSE_atom_tracker, val_forcesRMSE_tracker, val_forcesRMSE_atom_tracker]
best_val_energyRMSE = 1000.0
best_energyRMSE = 1000.0

# LR variables
best_watched_metric = 1000.0
lr_patience_counter = 0
lr_patience_max = 20

# Callbacks
learning_curves_logger = net_utils.LearningCurvesLogs(log_file=f"results/logs/{run_name}.json", run_name=run_name)

# Step functions
def train_on_energies(x, y):
  """Custom training step implemented in order to fit the network on the atom loss."""

  # count the number of atoms in order to compute the atom loss
  coordinates_batch, string_species_batch = x
  energies_batch = y
  num_species_batch = tf.cast(model.species_translator(string_species_batch), dtype=model.dtype)
  num_atoms = tf.math.count_nonzero(num_species_batch, axis=1, dtype=data_dtype)

  # Calling a model inside a GradientTape scope enables you to retrieve the gradients of the trainable weights of the layer
  # with respect to a loss value.
  # Inside the scope, we call the model (forward pass) and compute the loss we want to optimise.
  # Outside the scope, we retrieve and adjust the gradients of the weights of the model with regard to that loss.
  with tf.GradientTape() as step_tape:
    energies_pred = model(x, training=True)
    # Energy scalar loss
    energy_squared_errors = (energies_batch - energies_pred)**2
    energy_mse = tf.nn.compute_average_loss(energy_squared_errors, global_batch_size=batch_size)
    energy_atom_loss = tf.nn.compute_average_loss(  energy_squared_errors / tf.math.sqrt(num_atoms),
                                                    global_batch_size=batch_size )
  gradients_list = step_tape.gradient(energy_atom_loss, model.trainable_variables)
  model.optimizer.apply_gradients(zip(gradients_list, model.trainable_variables))
  # When you call apply_gradients within a distribution strategy scope, its behavior is modified:
  # before applying gradients on each parallel instance during synchronous training, it performs a sum-over-all-replicas of the gradients in order to aggregate them.

  return energy_mse, energy_atom_loss

def validate_on_energies(x, y):

  coordinates_batch, string_species_batch = x
  energies_batch = y
  num_species_batch = tf.cast(model.species_translator(string_species_batch), dtype=model.dtype)
  num_atoms = tf.math.count_nonzero(num_species_batch, axis=1, dtype=data_dtype)

  energies_pred = model(x, training=False)

  # Energy scalar loss
  energy_squared_errors = (energies_batch - energies_pred)**2
  energy_mse = tf.nn.compute_average_loss(energy_squared_errors, global_batch_size=batch_size)
  energy_atom_loss = tf.nn.compute_average_loss(  energy_squared_errors / tf.math.sqrt(num_atoms),
                                                  global_batch_size=batch_size )

  return energy_mse, energy_atom_loss

def train_on_energies_and_forces(x, y):
  """Implementation of the torchani loss function for forces (different from the one of the ani2x paper)."""

  coordinates_batch, string_species_batch = x
  energies_batch, forces_batch = y
  num_species_batch = tf.cast(model.species_translator(string_species_batch), dtype=model.dtype)
  num_atoms = tf.math.count_nonzero(num_species_batch, axis=1, dtype=data_dtype)

  # Computing the gradient
  with tf.GradientTape() as step_tape:      # GradientTape will automatically watch any trainable variables that are accessed inside the context
    with tf.GradientTape(watch_accessed_variables=False) as force_tape:
      force_tape.watch(coordinates_batch)
      energies_pred = model(x, training=True)
    forces_pred = -force_tape.gradient(energies_pred, coordinates_batch)
    # Energy scalar loss
    energy_squared_errors = (energies_batch - energies_pred)**2
    energy_mse = tf.nn.compute_average_loss(energy_squared_errors, global_batch_size=batch_size)
    energy_atom_loss = tf.nn.compute_average_loss(  energy_squared_errors / tf.math.sqrt(num_atoms),
                                                    global_batch_size=batch_size )
    # Forces scalar loss
    forces_squared_errors = (forces_batch - forces_pred)**2                             # (batch_size, num_atoms, 3)
    forces_squared_errors = tf.math.reduce_sum(forces_squared_errors, axis=[1,2])       # (batch_size,)
    forces_mse = tf.nn.compute_average_loss(forces_squared_errors, global_batch_size=batch_size)
    forces_atom_loss = tf.nn.compute_average_loss(  forces_squared_errors / num_atoms,          # only /num_atoms here, as in torchani
                                                    global_batch_size=batch_size )
    # Total loss
    total_loss = energy_atom_loss + 0.1*forces_atom_loss
  gradients_list = step_tape.gradient(total_loss, model.trainable_variables)
  
  # Applying the gradient
  model.optimizer.apply_gradients(zip(gradients_list, model.trainable_variables))

  return energy_mse, energy_atom_loss, forces_mse, forces_atom_loss

def validate_on_energies_and_forces(x, y):

  coordinates_batch, string_species_batch = x
  energies_batch, forces_batch = y
  num_species_batch = tf.cast(model.species_translator(string_species_batch), dtype=model.dtype)
  num_atoms = tf.math.count_nonzero(num_species_batch, axis=1, dtype=data_dtype)

  energies_pred, forces_pred = model.computeEnergyAndForces(x)

  # Energy scalar loss
  energy_squared_errors = (energies_batch - energies_pred)**2
  energy_mse = tf.nn.compute_average_loss(energy_squared_errors, global_batch_size=batch_size)
  energy_atom_loss = tf.nn.compute_average_loss(  energy_squared_errors / tf.math.sqrt(num_atoms),
                                                  global_batch_size=batch_size )
  # Forces scalar loss
  forces_squared_errors = (forces_batch - forces_pred)**2
  forces_squared_errors = tf.math.reduce_sum(forces_squared_errors, axis=[1,2])
  forces_mse = tf.nn.compute_average_loss(forces_squared_errors, global_batch_size=batch_size)
  forces_atom_loss = tf.nn.compute_average_loss(  forces_squared_errors / num_atoms,
                                                  global_batch_size=batch_size )

  return energy_mse, energy_atom_loss, forces_mse, forces_atom_loss

def max_abs(forces):
  return tf.reduce_max(tf.abs(forces))

@tf.function
def dist_train_on_energies(x, y):
  per_replica_losses = mirrored_strategy.run(train_on_energies, args=(x, y))
  return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)    # axis=None maintains the loss separated

@tf.function
def dist_validate_on_energies(x, y):
  per_replica_losses = mirrored_strategy.run(validate_on_energies, args=(x, y))
  return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def dist_train_on_energies_and_forces(x, y):
  per_replica_losses = mirrored_strategy.run(train_on_energies_and_forces, args=(x, y))
  return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def dist_validate_on_energies_and_forces(x, y):
  per_replica_losses = mirrored_strategy.run(validate_on_energies_and_forces, args=(x, y))
  return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def dist_max_abs(forces):
  per_replica_dummies = mirrored_strategy.run(max_abs, args=(forces,))
  return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_dummies, axis=None)   # SUM=MAX for our aim

#
##
### FIT!
##
#

# Datasets distribution
dist_training_set = mirrored_strategy.experimental_distribute_dataset(training_set)
dist_validation_set = mirrored_strategy.experimental_distribute_dataset(validation_set)

tf.print(f"\n** {run_name} started. Global batch size: {batch_size}. Number of GPUs: {num_gpus}. **")

epoch = last_epoch+1
while True:   # infinite number of epochs
  start_time = time.time()

  tf.print(f"\n** Epoch {epoch} started - lr: {model.optimizer.lr.numpy():.6e} **")

  ############################# TRAINING LOOP #####################################

  for x, y in dist_training_set:
    if dist_max_abs(y[1]) > 1e-6:    # if the forces are not dummies
      energy_mse, energy_atom_loss, forces_mse, forces_atom_loss = dist_train_on_energies_and_forces(x, y)
      forces_rmse = net_utils.hartree2kcalmol(tf.math.sqrt(forces_mse))       # these have to be computed outside the dist_train_step function beacuse of reduction policies
      forces_rmse_atom = net_utils.hartree2kcalmol(tf.math.sqrt(forces_atom_loss))
      forcesRMSE_tracker.update_state(forces_rmse)
      forcesRMSE_atom_tracker.update_state(forces_rmse_atom)
    else:
      y = y[0]    # we have to remove the dummy forces
      energy_mse, energy_atom_loss = dist_train_on_energies(x, y)
    energy_rmse = net_utils.hartree2kcalmol(tf.math.sqrt(energy_mse))
    energy_rmse_atom = net_utils.hartree2kcalmol(tf.math.sqrt(energy_atom_loss))
    energyRMSE_tracker.update_state(energy_rmse)
    energyRMSE_atom_tracker.update_state(energy_rmse_atom)

  # Save the model if this is a best energyRMSE epoch
  if energyRMSE_tracker.result() < best_energyRMSE:
    model.save(f"results/models/best_train/{run_name}_TL_{energyRMSE_tracker.result():.3f}")
    best_energyRMSE = energyRMSE_tracker.result()

  # Check if the learning rate has to be reduced
  watched_metric = energyRMSE_tracker.result() + 0.1*forcesRMSE_tracker.result()
  if watched_metric < (best_watched_metric-0.01):
    lr_patience_counter = 0
    best_watched_metric = watched_metric
  else:
    lr_patience_counter += 1
    if lr_patience_counter >= lr_patience_max:
      model.optimizer.lr /= 2
      if model.optimizer.lr < 1e-5:
        model.optimizer.lr = 1e-5     # to not go below 1e-5
      lr_patience_counter = 0

  tf.print(f"energyRMSE: {energyRMSE_tracker.result():.4f} - energyRMSE_atom: {energyRMSE_atom_tracker.result():.4f} - forcesRMSE: {forcesRMSE_tracker.result():.4f} - forcesRMSE_atom: {forcesRMSE_atom_tracker.result():.4f}")

  ############################# VALIDATION LOOP #####################################

  for x, y in dist_validation_set:
    if dist_max_abs(y[1]) > 1e-6:
      val_energy_mse, val_energy_atom_loss, val_forces_mse, val_forces_atom_loss = dist_validate_on_energies_and_forces(x, y)
      val_forces_rmse = net_utils.hartree2kcalmol(tf.math.sqrt(val_forces_mse))
      val_forces_rmse_atom = net_utils.hartree2kcalmol(tf.math.sqrt(val_forces_atom_loss))
      val_forcesRMSE_tracker.update_state(val_forces_rmse)
      val_forcesRMSE_atom_tracker.update_state(val_forces_rmse_atom)
    else:
      y = y[0]
      val_energy_mse, val_energy_atom_loss = dist_validate_on_energies(x, y)
    val_energy_rmse = net_utils.hartree2kcalmol(tf.math.sqrt(val_energy_mse))
    val_energy_rmse_atom = net_utils.hartree2kcalmol(tf.math.sqrt(val_energy_atom_loss))
    val_energyRMSE_tracker.update_state(val_energy_rmse)
    val_energyRMSE_atom_tracker.update_state(val_energy_rmse_atom)

  # Save the model if this is a best val_energyRMSE epoch
  if val_energyRMSE_tracker.result() < best_val_energyRMSE:
    model.save(f"results/models/best_val/{run_name}_VL_{val_energyRMSE_tracker.result():.3f}")
    best_val_energyRMSE = val_energyRMSE_tracker.result()

  tf.print(f"val_energyRMSE: {val_energyRMSE_tracker.result():.4f} - val_energyRMSE_atom: {val_energyRMSE_atom_tracker.result():.4f} - val_forcesRMSE: {val_forcesRMSE_tracker.result():.4f} - val_forcesRMSE_atom: {val_forcesRMSE_atom_tracker.result():.4f}")

  ###################################################################################

  # Log the metrics for this epoch
  learning_curves_logger.on_epoch_end(epoch-1, {m.name: float(m.result().numpy()) for m in metrics_list})

  # Save this model as the last one
  model.save(f"results/models/last_model_{run_name}")

  for m in metrics_list:
    m.reset_states()      # ok here because we have different train and validation trackers

  tf.print(f"** Epoch {epoch} finished - {time.time()-start_time:.2f}s **")

  epoch += 1