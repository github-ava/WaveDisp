import json
import os
from contextlib import redirect_stdout
from time import time
from typing import Optional, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class EpochProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time()
        time_taken = epoch_end_time - self.epoch_start_time
        log_message = f"Epoch {epoch + 1} time: {time_taken:.3f} s\n"

        # Check if the file exists, create it if not
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w"):
                pass  # Create an empty file

        # Append the log message to the file
        with open(self.file_path, "a") as file:
            file.write(log_message)


def positive_mean_squared_error(y_true, y_pred):
    # Calculate mean squared error
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    # Penalize negative predictions by squaring them
    mse_positive = tf.keras.losses.mean_squared_error(tf.zeros_like(y_pred), tf.maximum(0.0, y_pred - y_true))
    # Return the sum of both losses
    return mse + mse_positive


def custom_mean_squared_error(y_true, y_pred):
    # tf.print(y_true, output_stream=sys.stdout)
    return tf.reduce_mean(tf.square(y_true - y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    # tf.print(y_true, output_stream=sys.stdout)
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100


def train(PrNN: dict) -> tuple[tf.keras.Model, Any, Any, dict, np.ndarray]:
    """
    Train a neural network model based on the provided parameters.

    Parameters:
        PrNN (dict): Dictionary containing neural network training parameters.

    Returns:
        tuple[tf.keras.Model, Any, Any, dict, np.ndarray]:
            - Trained neural network model.
            - Input scaler used for normalization.
            - Output scaler used for normalization.
            - Model training history as a dictionary.
            - Array containing training, validation, and test losses.
    """
    output_data = np.loadtxt(f"{PrNN['in_dir']}samples")  # samples
    input_data = np.loadtxt(f"{PrNN['in_dir']}cp_values")  # cp_values

    if PrNN['num_samples'] > 0:
        output_data = output_data[:PrNN['num_samples'], :]
        input_data = input_data[:PrNN['num_samples'], :]

    n_s = input_data.shape[0]  # num samples
    n_w = input_data.shape[1]  # length of input (effective curve-frequency)
    n_p = output_data.shape[1]  # num output param (Cs+H)

    # output_data[:, :num_layer + 1] = output_data[:, :num_layer + 1] / 100.

    # Normalize data
    if PrNN['scaling'] == 0:
        scaler_input = MinMaxScaler()
        scaler_output = MinMaxScaler()
    else:
        scaler_input, scaler_output = None, None
    input_data_normalized = scaler_input.fit_transform(input_data)
    output_data_normalized = scaler_output.fit_transform(output_data)

    # Split data into training, validation, and testing sets
    shuffle_data = PrNN['shuffle_data']
    save_period = PrNN['save_period']
    patience = PrNN['patience']

    if PrNN['activation'] == 0:
        activation = 'relu'
    elif PrNN['activation'] == 1:
        activation = 'sigmoid'
    elif PrNN['activation'] == 2:
        activation = tf.keras.layers.LeakyReLU()
    elif PrNN['activation'] == 3:
        activation = tf.keras.layers.LeakyReLU(alpha=0.1)  # negative_slope
    else:
        activation = None

    if PrNN['initializer'] == 0:
        initializer = 'he_normal'
    elif PrNN['initializer'] == 1:
        initializer = 'glorot_normal'
    elif PrNN['initializer'] == 2:
        initializer = 'random_normal'
    else:
        initializer = None

    learning_rate = PrNN['learning_rate']

    if PrNN['optimizer'] == 0:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif PrNN['optimizer'] == 1:
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        optimizer = None

    batch_size = PrNN['batch_size']
    train_ratio = PrNN['train_ratio']
    val_ratio = PrNN['val_ratio']
    train_samples = int(train_ratio * n_s)
    val_samples = int(val_ratio * n_s)

    save_freq_batches = int(save_period * max(n_s / batch_size, 1))

    if shuffle_data:
        indices = np.random.permutation(n_s)
        input_data_normalized = input_data_normalized[indices]
        output_data_normalized = output_data_normalized[indices]

    train_input = input_data_normalized[:train_samples]
    train_output = output_data_normalized[:train_samples]
    val_input = input_data_normalized[train_samples:train_samples + val_samples]
    val_output = output_data_normalized[train_samples:train_samples + val_samples]
    test_input = input_data_normalized[train_samples + val_samples:]
    test_output = output_data_normalized[train_samples + val_samples:]

    # Build model
    if PrNN['model'] == 0:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_w,)),
            tf.keras.layers.Dense(units=n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_p + n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_p, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_p)
        ])
    elif PrNN['model'] == 1:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_w,)),
            tf.keras.layers.Reshape((n_w, 1)),  # Reshape input for 1D convolution
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation=activation),  # Convolutional layer
            tf.keras.layers.MaxPooling1D(pool_size=2),  # Max pooling layer
            tf.keras.layers.Flatten(),  # Flatten for fully connected layers
            tf.keras.layers.Dense(units=n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_p + n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_p + n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_p, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=n_p)
        ])

    elif PrNN['model'] == 2:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_w,)),
            tf.keras.layers.Dense(units=n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.Dense(units=n_p + n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=n_w, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.Dense(units=n_p)
        ])
    else:
        model = None

    # model.summary()

    if PrNN['loss'] == 0:
        loss = tf.keras.losses.MeanSquaredError()
    elif PrNN['loss'] == 1:
        loss = tf.keras.losses.MeanSquaredLogarithmicError()
    elif PrNN['loss'] == 2:
        loss = tf.keras.losses.MeanAbsoluteError()
    elif PrNN['loss'] == 3:
        loss = tf.keras.losses.Huber()
    else:
        loss = None

    if PrNN['metrics'] == 0:  # metric to print during training
        metrics = ['mean_squared_error']
    else:
        metrics = None

    if PrNN['monitor'] == 0:
        monitor = 'val_loss'
    else:
        monitor = None

    # Compile model
    model.compile(optimizer=optimizer,
                  # mean_squared_error, mean_squared_logarithmic_error
                  # positive_mean_squared_error, custom_mean_squared_error
                  loss=loss,
                  metrics=metrics)  # [mean_absolute_percentage_error]) - # metric to print during training

    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,  # 'custom_mean_squared_error'
                                                      # 'loss','accuracy','val_loss','val_accuracy'.
                                                      patience=patience)

    # Define checkpoint callback
    checkpoint_path = PrNN['out_dir'] + "model_checkpoint_epoch_{epoch:04d}.weights.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                                             save_freq=save_freq_batches)

    # Train model
    t = time()

    epoch_progress_path = f"{PrNN['out_dir']}model_epoch_progress.txt"
    with open(epoch_progress_path, "w"):
        pass
    epoch_progress_callback = EpochProgressCallback(epoch_progress_path)

    with open(f"{PrNN['out_dir']}model_fit_output.txt", 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            model_history = model.fit(train_input, train_output,
                                      epochs=PrNN['max_epochs'],
                                      batch_size=batch_size,
                                      validation_data=(val_input, val_output),
                                      callbacks=[early_stopping, checkpoint_callback, epoch_progress_callback],
                                      verbose=PrNN['verbose'])
    if PrNN['expr_id'] == 0:
        print('Training time: {:.3f} s'.format(time() - t))

    # Evaluate model
    with open(f"{PrNN['out_dir']}model_fit_output.txt", 'a', encoding='utf-8') as f:
        f.write('\n\nTraining time: {:.3f} s\n\n'.format(time() - t))
        with redirect_stdout(f):
            test_eval_output = model.evaluate(test_input, test_output)

    # Save model
    if PrNN['save_model'] is True:
        model.save(f"{PrNN['out_dir']}trained_model.keras")
        joblib.dump(scaler_input, f"{PrNN['out_dir']}scaler_input.joblib")
        joblib.dump(scaler_output, f"{PrNN['out_dir']}scaler_output.joblib")
        json.dump(model_history.history, open(f"{PrNN['out_dir']}trained_model.json", 'w'))

    # metric_names = model_history.history.keys()
    # print("Available metrics:", metric_names)

    test_eval_output = test_eval_output if isinstance(test_eval_output, (tuple, list)) else (test_eval_output,)
    train_valid_test_loss = np.array(
        [model_history.history['loss'][-1], model_history.history['val_loss'][-1], test_eval_output[0]])
    print(f"Train, Valid, Test Loss ({PrNN['expr_id']}): {str(train_valid_test_loss)}")
    return model, scaler_input, scaler_output, model_history.history, train_valid_test_loss


def history(PrNN: dict, model_history: Optional[dict] = None) -> None:
    """
    Plot the training history of a neural network model.

    Parameters:
        PrNN (dict): Dictionary containing neural network training parameters.
        model_history (Optional[dict]): Optional argument to provide the model training history directly.
                                        If not provided, it is loaded from a file based on PrNN['out_dir'].

    Returns:
        None
    """

    if PrNN['save_model'] is True:
        model_history = json.load(open(f"{PrNN['out_dir']}trained_model.json", 'r'))

    # Plot training history
    plt.clf()
    plt.plot(np.log10(model_history['loss']), label='Training Loss')
    plt.plot(np.log10(model_history['val_loss']),  # model_history['val_custom_mean_squared_error'])
             label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log10(Loss)')
    plt.legend()
    plt.savefig(f"{PrNN['out_dir']}training_history.png")
    if PrNN['show_plots']:
        plt.show()


def prediction(PrNN: dict, input_array: np.ndarray, model: Optional[tf.keras.Model] = None,
               scaler_input: Optional[joblib.load] = None, scaler_output: Optional[joblib.load] = None) -> np.ndarray:
    """
    Predicts the output using the provided input array and trained neural network model.

    Parameters:
    - PrNN (dict): Dictionary containing parameters for the neural network.
    - input_array (np.ndarray): Input array for prediction.
    - model (Optional[tf.keras.Model]): Trained neural network model. If not provided, it will be loaded from disk.
    - scaler_input (Optional[joblib.load]): Scaler for input normalization. If not provided, it will be loaded from disk.
    - scaler_output (Optional[joblib.load]): Scaler for output denormalization. If not provided, it will be loaded from disk.

    Returns:
    - np.ndarray: Predicted output array.
    """
    if input_array.shape[0] == 1:
        input_array = np.array(input_array).reshape(1, -1)
    if PrNN['save_model'] is True:
        model = tf.keras.models.load_model(f"{PrNN['out_dir']}trained_model.keras")
        # model = tf.keras.models.load_model(f'{out_dir}trained_model',
        #                                  custom_objects={'positive_mean_squared_error': positive_mean_squared_error})
        # model.load_weights("model_checkpoint_epoch_0150.weights.h5")
        scaler_input = joblib.load(f"{PrNN['out_dir']}scaler_input.joblib")
        scaler_output = joblib.load(f"{PrNN['out_dir']}scaler_output.joblib")

    normalized_input_array = scaler_input.transform(input_array)
    with open(f"{PrNN['out_dir']}model_fit_output.txt", 'a', encoding='utf-8') as f:
        with redirect_stdout(f):
            predicted_output_normalized = model.predict(normalized_input_array)
    predicted_output = scaler_output.inverse_transform(predicted_output_normalized)
    # predicted_output[0, :num_layer + 1] = predicted_output[0, :num_layer + 1] * 100.
    return predicted_output
