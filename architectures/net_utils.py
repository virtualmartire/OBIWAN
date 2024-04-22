import tensorflow as tf
import json

from architectures import anakin, w_anakin, w_anakin_oe, w_anakin_oe_full, obiwan

#
##
### METRICS
##
#

def hartree2kcalmol(x):
    # Hartree to kcal/mol conversion factor from CODATA 2014
    EV_TO_JOULE = 1.6021766208e-19
    HARTREE_TO_EV = 27.211386024367243
    JOULE_TO_KCAL = 1 / 4184.
    AVOGADROS_NUMBER = 6.022140857e+23
    HARTREE_TO_JOULE = HARTREE_TO_EV * EV_TO_JOULE
    HARTREE_TO_KCALMOL = HARTREE_TO_JOULE * JOULE_TO_KCAL * AVOGADROS_NUMBER
    return x * HARTREE_TO_KCALMOL

def kcalmolRMSE(y_true, y_pred):
    return hartree2kcalmol(tf.math.sqrt(tf.keras.metrics.mean_squared_error(y_true, y_pred)))

#
##
### CALLBACKS
##
#

class LearningCurvesLogs(tf.keras.callbacks.Callback):
    """Callback that records losses and metrics in the LearningCurves format."""

    def __init__(self, log_file, run_name):
        super().__init__()

        self.log_file = log_file
        self.run_name = run_name

        try:        # see if the file already exists
            with open(self.log_file, "r") as f:
                self.json_records = json.load(f)
            try:    # see if this is a resume training
                self.history = self.json_records[self.run_name]
            except:
                self.history = {}
        except:
            self.json_records = {}
            self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        """WARNING: epoch is 0-indexed!"""

        logs = logs or {}
        for performance_name, performance_value in logs.items():
            try:        # try to overwrite first (and delete all the now-old future values in case)
                self.history[performance_name][epoch] = performance_value
                self.history[performance_name] = self.history[performance_name][:epoch+1]
            except:     # if the performance doesn't already exist or if this is a completely new epoch
                self.history.setdefault(performance_name, [None for _ in range(epoch)]).append(performance_value)
        
        with open(self.log_file, "w") as f:
            self.json_records[self.run_name] = self.history
            json.dump(self.json_records, f)

        return
    
#
##
### UTILITY FUNCTIONS
##
#

def is_not_toxic(model, x, y):
    """Filter function that says if a molecule has NOT NaN second derivatives (due to a TensorFlow bug in case)."""

    coordinates_batch, string_species_batch = x
    energies_batch, forces_batch = y
    num_species_batch = tf.cast(model.species_translator(string_species_batch), dtype=model.dtype)
    num_atoms = tf.math.count_nonzero(num_species_batch, axis=1, dtype=coordinates_batch.dtype)

    with tf.GradientTape() as step_tape:
        # Energy scalar loss
        with tf.GradientTape(watch_accessed_variables=False) as force_tape:
            force_tape.watch(coordinates_batch)
            energies_pred = model(x)
        forces_pred = -force_tape.gradient(energies_pred, coordinates_batch)
        energy_squared_errors = (energies_batch - energies_pred)**2
        energy_atom_loss = tf.nn.compute_average_loss(  energy_squared_errors / tf.math.sqrt(num_atoms),
                                                        global_batch_size=1 )
        # Forces scalar loss
        forces_squared_errors = (forces_batch - forces_pred)**2                             # (batch_size, num_atoms, 3)
        forces_squared_errors = tf.math.reduce_sum(forces_squared_errors, axis=[1,2])       # (batch_size,)
        forces_atom_loss = tf.nn.compute_average_loss(  forces_squared_errors / num_atoms,          # only /num_atoms here, as in torchani
                                                        global_batch_size=1 )
        # Total loss
        total_loss = energy_atom_loss + 0.1*forces_atom_loss
    gradients_list = step_tape.gradient(total_loss, model.trainable_variables)

    any_nan_list = [tf.math.reduce_any(tf.math.is_nan(grad)) for grad in gradients_list]
    any_nan = tf.math.reduce_any(any_nan_list)

    return tf.math.logical_not(any_nan)

def getModel(model_name, **kwargs):
    if model_name == "anakin":
        return anakin.Anakin(**kwargs)
    elif model_name == "w_anakin":
        return w_anakin.WAnakin(**kwargs)
    elif model_name == "w_anakin_oe":
        return w_anakin_oe.WAnakinOE(**kwargs)
    elif model_name == "w_anakin_oe_full":
        return w_anakin_oe_full.WAnakinOEFull(**kwargs)
    elif model_name == "obiwan":
        return obiwan.Obiwan(**kwargs)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")