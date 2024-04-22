import numpy as np
import tensorflow as tf
import h5py
from rdkit import Chem
import os
import torch

import data.pyanitools as pya


#
##
### UTILS
##
#


def getTfDtype(dtype_name):
  if dtype_name == 'float32':
    return tf.float32
  elif dtype_name == 'float64':
    return tf.float64
  else:
    raise Exception(f"Unknown dtype: {dtype_name}")

def atomicNumbersToStrings(scalar):
  if scalar == 0:
    return ""
  elif scalar == 1:
    return "H"
  elif scalar == 6:
    return "C"
  elif scalar == 7:
    return "N"
  elif scalar == 8:
    return "O"
  elif scalar == 9:
    return "F"
  elif scalar == 15:
    return "P"
  elif scalar == 16:
    return "S"
  elif scalar == 17:
    return "Cl"
  elif scalar == 35:
    return "Br"
  elif scalar == 53:
    return "I"
  else:
    raise Exception(f"Unknown atomic number found: {scalar}")

atomicNumbersToStrings = np.frompyfunc(atomicNumbersToStrings, 1, 1)

def areThereChargedAtoms(smiles):

  mol = Chem.MolFromSmiles(smiles)

  for atom in mol.GetAtoms():
    if atom.GetFormalCharge() != 0:
      return True
    else:
      continue
    
  return False

def bohrToAngstrom(array):
  return array * 0.529177

def hartreePerBohrToHartreePerAngstrom(array):
  return array * 1.88973


#
##
### GENERATORS
##
#


def ani1_generator():

  anidataloader_object = pya.anidataloader("data/datasets/ani1x/ani1x-release.h5")

  # CONFIGURATION = atomic species and bonds involved
  # CONFORMATION = coordinates of the atoms
  for molecule_configuration in anidataloader_object:

      coordinates = molecule_configuration['coordinates']             # coordinates of all the conformations of the molecule
      atomic_numbers = molecule_configuration['atomic_numbers']
      species = atomicNumbersToStrings(atomic_numbers)

      energies = molecule_configuration['wb97x_dz.energy']
      forces = molecule_configuration['wb97x_dz.forces']

      for conformation, energy, forces_ in zip(coordinates, energies, forces):
          yield (conformation, species), (energy, forces_)

  anidataloader_object.cleanup()

def ani2_generator():

  anidataloader_object = pya.anidataloader("data/datasets/ani2x/ANI-2x-wB97X-631Gd.h5")

  # CONFIGURATION = atomic species and bonds involved
  # CONFORMATION = coordinates of the atoms
  for molecule_configuration in anidataloader_object:

      coordinates = molecule_configuration['coordinates']
      atomic_numbers = molecule_configuration['species']
      species = atomicNumbersToStrings(atomic_numbers)

      energies = molecule_configuration['energies']
      forces = molecule_configuration['forces']

      for conformation, species_, energy, forces_ in zip(coordinates, species, energies, forces):
          yield (conformation, species_), (energy, forces_)

  anidataloader_object.cleanup()

def spice_generator():

  h5_dataset = h5py.File("data/datasets/spice/SPICE.hdf5", 'r')

  for molecule_id in h5_dataset:    # 1 molecule_id = 1 smiles = 1 configuration

    molecule_smiles = h5_dataset[molecule_id]['smiles'][0]

    if areThereChargedAtoms(molecule_smiles) == False:        # we don't want molecules with charged atoms

      atomic_numbers = h5_dataset[molecule_id]['atomic_numbers']
      species = atomicNumbersToStrings(atomic_numbers)

      conformations = h5_dataset[molecule_id]['conformations']
      energies = h5_dataset[molecule_id]['dft_total_energy']
      forces = h5_dataset[molecule_id]['dft_total_gradient']

      for conformation, energy, forces_ in zip(conformations, energies, forces):    # there is only one species vector for each molecule
        yield (bohrToAngstrom(conformation), species), (energy, hartreePerBohrToHartreePerAngstrom(forces_))
    
    else:
      continue

  h5_dataset.close()


#
##
### TENSORFLOW UTILS
##
#

def loadShuffleSplit(chosen_dataset, data_dtype):

  if chosen_dataset == 'ani1x':
    cardinality = 4_956_005
    dataset = tf.data.Dataset.from_generator(   ani1_generator,
                                                output_signature = (
                                                                    (
                                                                    tf.TensorSpec(shape=(None, 3), dtype=data_dtype),
                                                                    tf.TensorSpec(shape=(None,), dtype=tf.string)
                                                                    ),
                                                                    (
                                                                    tf.TensorSpec(shape=(), dtype=data_dtype),
                                                                    tf.TensorSpec(shape=(None, 3), dtype=data_dtype),
                                                                    )
                                                                )
                                                )
      
  elif chosen_dataset == 'ani2x':
    cardinality = 4_695_707
    dataset = tf.data.Dataset.from_generator(   ani2_generator,
                                                output_signature = (
                                                                    (
                                                                    tf.TensorSpec(shape=(None, 3), dtype=data_dtype),
                                                                    tf.TensorSpec(shape=(None,), dtype=tf.string)
                                                                    ),
                                                                    (
                                                                    tf.TensorSpec(shape=(), dtype=data_dtype),
                                                                    tf.TensorSpec(shape=(None, 3), dtype=data_dtype),
                                                                    )
                                                                )
                                                )
      
  elif chosen_dataset == 'spice':
    cardinality = 627_692
    dataset = tf.data.Dataset.from_generator(   spice_generator,
                                                output_signature = (
                                                                    (
                                                                    tf.TensorSpec(shape=(None, 3), dtype=data_dtype),
                                                                    tf.TensorSpec(shape=(None,), dtype=tf.string)
                                                                    ),
                                                                    (
                                                                    tf.TensorSpec(shape=(), dtype=data_dtype),
                                                                    tf.TensorSpec(shape=(None, 3), dtype=data_dtype),
                                                                    )
                                                                )
                                                )
    
  else:
    raise Exception(f"Unknown dataset: {chosen_dataset}")

  # Shuffle
  dataset = dataset.shuffle(cardinality, seed=777, reshuffle_each_iteration=False)

  # Split
  train_val_split = 0.8
  num_train_batches = int(cardinality * train_val_split)
  training_set = dataset.take(num_train_batches)
  validation_set = dataset.skip(num_train_batches)

  return training_set, validation_set

def getDatasets(ds_name, data_dtype_name="float32"):

  data_dtype = getTfDtype(data_dtype_name)

  training_set, validation_set = loadShuffleSplit(ds_name, data_dtype)

  training_set_path = f"data/datasets/{ds_name}/cache/{data_dtype_name}/training/"
  os.makedirs(training_set_path, exist_ok=True)
  training_set = training_set.cache(training_set_path)

  validation_set_path = f"data/datasets/{ds_name}/cache/{data_dtype_name}/validation/"
  os.makedirs(validation_set_path, exist_ok=True)
  validation_set = validation_set.cache(validation_set_path)

  return training_set, validation_set

#
##
### COMP6 utils
##
#

def COMP6v2Yielder(path):

  anidataloader_object = pya.anidataloader(path)

  for molecule_configuration in anidataloader_object:

    coordinates = molecule_configuration['coordinates']
    energies = molecule_configuration['energies']

    atomic_numbers = molecule_configuration['species']
    species = atomicNumbersToStrings(atomic_numbers)

    for conformation, energy, species_ in zip(coordinates, energies, species):
      yield (conformation, species_), energy

  anidataloader_object.cleanup()


def getBatchedDataset(ds_path, batch_size, data_dtype=tf.float32):

  ds_name = "-".join(ds_path.split("/")[-1].split(".")[0].split("-")[0:2])

  # Load
  dataset = tf.data.Dataset.from_generator(   lambda: COMP6v2Yielder(ds_path),
                                              output_signature = (
                                                                  (
                                                                  tf.TensorSpec(shape=(None, 3), dtype=data_dtype),
                                                                  tf.TensorSpec(shape=(None,), dtype=tf.string)
                                                                  ),
                                                                  tf.TensorSpec(shape=(), dtype=data_dtype)
                                                              )
                                                          )
  
  # Batch
  if ds_name == "ANI-BenchMD":
    dataset = dataset.padded_batch(batch_size=1, padding_values=((tf.constant(1e20, dtype=data_dtype), ""), None), drop_remainder=False)
  else:
    dataset = dataset.padded_batch(batch_size=batch_size, padding_values=((tf.constant(1e20, dtype=data_dtype), ""), None), drop_remainder=False)

  return dataset


def OBITestOnCOMP6v2(ds_path, model, batch_size=64):

  @tf.function
  def compiledCall(inputs):
    return model(inputs)

  # Get the test set
  tf_dataset = getBatchedDataset(ds_path=ds_path, batch_size=batch_size)

  # Predict
  predictions_array = np.array([])
  labels_array = np.array([])
  for inputs_batch, energies_batch in tf_dataset:
    predictions = compiledCall(inputs_batch)
    predictions_array = np.concatenate([predictions_array, predictions.numpy().flatten()])
    labels_array = np.concatenate([labels_array, energies_batch.numpy().flatten()])

  # Compute the squared errors
  squared_errors = (predictions_array - labels_array)**2

  return squared_errors.tolist()


def ANITestOnCOMP6v2(ds_path, model):

  anidataloader_object = pya.anidataloader(ds_path)
  predictions_array = np.array([])
  labels_array = np.array([])

  for molecule_configuration in anidataloader_object:

    atomic_numbers_batch = molecule_configuration['species']
    atomic_numbers_batch = torch.from_numpy(atomic_numbers_batch)

    coordinates_batch = molecule_configuration['coordinates']
    coordinates_batch = torch.from_numpy(coordinates_batch)

    energies_batch = molecule_configuration['energies']
    energies_batch = torch.from_numpy(energies_batch)

    predictions = model((atomic_numbers_batch, coordinates_batch)).energies.detach()
    predictions_array = np.concatenate([predictions_array, predictions.numpy().flatten()])
    labels_array = np.concatenate([labels_array, energies_batch.numpy().flatten()])

  squared_errors = (predictions_array - labels_array)**2

  anidataloader_object.cleanup()

  return squared_errors.tolist()
