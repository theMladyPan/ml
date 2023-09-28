#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# example document
# {
#     "_id": {
#       "$oid": "6515c5aa38f7d3b3223eab5a"
#     },
#     "measurements": {
#       "4": {
#         "pv0": 0.393014073,
#         "pv1": 0.399152368,
#         "pv2": 0.232258141,
#         "pv3": 0.232182011
#       },
#       "20": {
#         "pv0": -0.0625,
#         "pv1": 22.0625,
#         "pv2": 20.625,
#         "pv3": 0
#       }
#     },
#     "time": {
#       "server": {
#         "epoch": 1695925674.8435066
#       }
#     }
#   },

import os
from pymongo.mongo_client import MongoClient
import logging
import pickle
import argparse
import dotenv
import numpy as np

# Load environment variables
dotenv.load_dotenv()

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument(
    '-r', 
    '--reload', 
    help='Reload data from the server', 
    required=False, 
    default=False, 
    action='store_true'
)
parser.add_argument(
    "-n",
    "--number",
    help="Number of sensors to predict",
    default=8,
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

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

log.info(f"Current working directory: {os.getcwd()}")
sensor_file_name = f"data/sensor{args.number}.pkl"

filter = {
    "$and": [
        {"measurements.20" : {"$exists": True}},
        {f"measurements.{args.number}" : {"$exists": True}}
    ]
}
projection = { 
    f"measurements.{args.number}.pv0": 1,
    f"measurements.{args.number}.pv1": 1,
    "measurements.20.pv1": 1,
    "measurements.20.pv2": 1,
    "time.server.epoch": 1
}

# check if file data/sensor8.pkl exists
if os.path.exists(sensor_file_name) and not args.reload:
    log.info('File exists, loading data from file')
    data = pickle.load(open(sensor_file_name, 'rb'))
else:
    log.info('File does not exist, loading data from server')
    uri = os.getenv('MONGO_URI')
    # Create a new client and connect to the server
    client = MongoClient(uri)
    # Access the database
    db = client['sim-bridge']
    # Access the collection
    collection = db['PRJ-7']
    # Get all documents
    documents = collection.find(filter, projection)
    count = collection.count_documents(filter)
    # Create a new list
    data = []
    it = 1
    # Loop through all documents
    for document in documents:
        # Append the document to the list
        print(f"Downloading document: {it}/{count}", end="\r")
        it += 1
        data.append(document)
    print()
    # Save the data to a file
    with open(sensor_file_name, 'wb') as f:
        pickle.dump(data, f)
    log.info('File saved')
    # Close the connection
    client.close()

log.info(f'Data loaded, length: {len(data)}')

# extract sensor N data

sensor_data = np.array([
    document["measurements"][str(args.number)]["pv0"] - document["measurements"][str(args.number)]["pv1"]
 for document in data])

temp_data = np.array([[
    document["measurements"]["20"]["pv1"],
    document["measurements"]["20"]["pv2"]
] for document in data])

timestamp_data = np.array([document["time"]["server"]["epoch"] for document in data])

import tensorflow as tf
import tensorboard
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers, models


SEQUENCE_LENGTH = args.sequence
# Define your model
model = models.Sequential()

# Add the input layer with two features: time and temperature difference
model.add(layers.Input(shape=(SEQUENCE_LENGTH, 2)))  # Allow for variable-length sequences

# Add an LSTM layer to capture sequential dependencies
model.add(layers.LSTM(1024, activation='relu', return_sequences=True))  # Return sequences for further processing

# Add one or more dense layers after the LSTM
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(64, activation='relu'))

# Add the output layer with one neuron to predict length difference
model.add(layers.Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print a summary of the model's architecture
model.summary()

# Define the Keras TensorBoard callback.
log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

TRAIN_RATIO = 0.8

# Function to create sequences from your data
def create_sequences(data, SEQUENCE_LENGTH):
    sequences = []
    for i in range(len(data) - SEQUENCE_LENGTH + 1):
        sequence = data[i:i + SEQUENCE_LENGTH]
        sequences.append(sequence)
    return np.array(sequences)

# Reshape the data into sequences
sensor_data_seq = create_sequences(sensor_data, SEQUENCE_LENGTH)
temp_data_seq = create_sequences(temp_data, SEQUENCE_LENGTH)

# Split the data into training and test sets
split_index = int(TRAIN_RATIO * len(sensor_data_seq))
X_train_seq = temp_data_seq[:split_index]
y_train_seq = sensor_data_seq[:split_index]
X_test_seq = temp_data_seq[split_index:]
y_test_seq = sensor_data_seq[split_index:]
timestamp_train = timestamp_data[:split_index]
timestamp_test = timestamp_data[split_index + SEQUENCE_LENGTH - 1:]

# Train the model.
model.fit(
    X_train_seq,
    y_train_seq, 
    batch_size=64,
    epochs=args.epochs, 
    callbacks=[tensorboard_callback])

# Save the model
model.save(f"models/sensor{args.number}.h5")

prediction1 = model.predict(X_train_seq)
prediction2 = model.predict(X_test_seq)

import matplotlib.pyplot as plt

plt.plot(timestamp_train, y_train_seq[:, 0], label="Training")
plt.plot(timestamp_train, prediction1[:, 0], label="Predicted")
plt.plot(timestamp_test, y_test_seq[:, 0], label="Actual")
plt.plot(timestamp_test, prediction2[:, 0], label="Predicted")
plt.legend()
plt.show()

