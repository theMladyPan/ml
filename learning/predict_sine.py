#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import argparse
import dotenv
import numpy as np
import math
import matplotlib.pyplot as plt


# Load environment variables
dotenv.load_dotenv()

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument(    
    "-n",
    "--n_data",
    help="Number of data points",
    default=1000,
    type=int
)
parser.add_argument(
    "-s",
    "--sequence",
    help="Length of the sequence",
    default=10,
    type=int
)
parser.add_argument(
    "-e",
    "--epochs",
    help="Number of epochs",
    default=5,
    type=int
)
parser.add_argument(
    "-t",
    "--retrain",
    help="Retrain the model",
    default=False,
    action='store_true'
)
parser.add_argument(
    "-d",
    "--density",
    help="Density of the model",
    default=512,
    type=int
)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SCRIPT_LOCATION = os.path.dirname(os.path.realpath(__file__))
os.chdir(SCRIPT_LOCATION)
log.info(f"Current working directory: {os.getcwd()}")

SIZE = args.n_data

def square_approx(x):
    return (4/np.pi) * (np.sin(x) + (1/3) * np.sin(3*x) + (1/5) * np.sin(5*x)) * (np.exp(-x / 20))

# create timestamp 0-1000
timestamps = np.arange(0, SIZE, 1)

# create data
data = square_approx(timestamps / 50)
# add noise to data
data = data + np.random.normal(0, 0.01, len(data))
time_sine = np.sin(timestamps / 50)
train_data = np.array([[
        timestamps[i], 
        time_sine[i]
    ] for i in range(len(timestamps))]
)

# Split the data into training and test sets
TRAIN_RATIO = 0.8
SEQUENCE_LENGTH = args.sequence
SPLIT_POINT = int(TRAIN_RATIO * SIZE)

# Function to create sequences from your data
def create_sequences(data, SEQUENCE_LENGTH):
    sequences = []
    for i in range(len(data) - SEQUENCE_LENGTH + 1):
        sequence = data[i:i + SEQUENCE_LENGTH]
        sequences.append(sequence)
    return np.array(sequences)


# Reshape the data into sequences
y_data_seq = create_sequences(data, SEQUENCE_LENGTH)
X_data_seq = create_sequences(train_data, SEQUENCE_LENGTH)

log.info("Splitting data...")
# Split the data into training and test sets
split_index = int(TRAIN_RATIO * len(y_data_seq))
X_train_seq = X_data_seq[:split_index]
y_train_seq = y_data_seq[:split_index]
X_test_seq = X_data_seq[split_index:]
y_test_seq = y_data_seq[split_index:]

import tensorboard
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# check if model exists
if os.path.exists(f"models/sine.h5") and not args.retrain:
    log.info('Model exists, loading model from file')
    model = keras.models.load_model(f"models/sine.h5")

else:
    log.info('Model does not exist, creating model')

    # Define your model
    model = models.Sequential()

    # Add the input layer with two features: time_sine, time_cosine and temperature difference
    model.add(layers.Input(shape=(SEQUENCE_LENGTH, len(X_train_seq[0][0]))))

    # Add an LSTM layer to capture sequential dependencies
    model.add(layers.LSTM(args.density, activation='tanh', return_sequences=True))  # Return sequences for further processing

    # Add one or more dense layers after the LSTM
    model.add(layers.Dense(args.density, activation='relu'))

    # Add the output layer with one neuron to predict length difference
    model.add(layers.Dense(1, activation='linear'))

    log.info("Compiling model")
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='mean_squared_error'
    )

    log.info("Model summary:")
    # Print a summary of the model's architecture
    model.summary()
    # Define the Keras TensorBoard callback.
    log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='loss',
        min_delta=0.0,
        patience=100,
        mode='min',
        verbose=1,
        restore_best_weights=True,
        start_from_epoch=10
    )

    log.info("Training model...")
    # Train the model.
    model.fit(
        X_train_seq,
        y_train_seq,
        batch_size=32,
        epochs=args.epochs,
        callbacks=[tensorboard_callback, early_stopping])
    
    log.info("Saving model...")
    model.save("models/sine.h5")

model_evaluation_loss = model.evaluate(X_test_seq, y_test_seq)

# Save the model
log.info("Predicting...")
prediction = model.predict(X_test_seq)
# do just one prediction:
prediction2 = model.predict(np.array([X_test_seq[0]]))

print(np.array([X_test_seq[0]]))
print(prediction2, y_test_seq[0])

log.info(f"Model evaluation on test data: {model_evaluation_loss}")

# Plot the results
log.info("Plotting results...")


plt.figure(figsize=(16, 12))
plt.plot(timestamps[:SPLIT_POINT], data[:SPLIT_POINT], label="Train Data")
plt.plot(timestamps[SPLIT_POINT:], data[SPLIT_POINT:], label="Test Data", alpha=0.5)

plt.plot(timestamps[SPLIT_POINT + 1:], prediction[:, -1], label="Prediction")
# for i in range(SEQUENCE_LENGTH):
#     plt.plot(timestamps[int(len(timestamps) * TRAIN_RATIO) + 1:], prediction[:, i], label=f"Prediction {i}")
plt.legend()
try:
    plt.savefig('function_approximation.png')
except PermissionError:
    log.warning("Could not save figure")

plt.show()
