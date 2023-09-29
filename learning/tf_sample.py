import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import logging
import math
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

def weierstrass(x):
    result = 0.0
    a = 0.5  # Amplitude
    b = 3.0  # Frequency parameter
    n_max = 100  # Number of terms in the series

    for n in range(n_max):
        result += a**n * np.cos(b**n * np.pi * x)

    return result

def random_combined_function(x):
    # Polynomial Function
    poly_result = 2 * x**3 - 3 * x**2 + 4 * x - 1

    # Trigonometric Function
    trig_result = np.sin(2 * x) + np.cos(x)

    # Exponential Decay
    decay_result = np.exp(-0.2 * x) * np.cos(3 * x)

    # Logistic Growth
    growth_result = 1 / (1 + np.exp(-0.1 * x))

    # Combine the results (e.g., taking the average)
    combined_result = ((poly_result + growth_result - trig_result) * decay_result)

    return combined_result

# Generate random data points and calculate their sine values
np.random.seed(0)

data_size = 2000
train_ratio = 0.8
split_index = int(data_size * train_ratio)

X = np.random.rand(data_size) * 2 * np.pi
y = np.array([weierstrass(x) for x in X])

# select 80 random X points for training 
X_train = np.array([[
    x,
    math.cos(x)
] for x in X[:split_index]])
y_train = np.array([weierstrass(x) for x in X[:split_index]])

# select 20 random X points for testing
X_test = np.array([[
    x,
    math.cos(x)
] for x in X[split_index:]])
y_test = np.array([weierstrass(x) for x in X[split_index:]])

# Define your model
model = models.Sequential()

# Add the input layer with two features: time_sine, time_cosine and temperature difference
model.add(layers.Input(shape=(2)))  # Allow for variable-length sequences
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(2**14, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(256, activation='relu'))

# Add the output layer with one neuron to predict length difference
model.add(layers.Dense(1, activation='linear'))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error'
)

# Print a summary of the model's architecture
model.summary()

# Define early stopping
early_stopping = EarlyStopping(
    monitor='loss',
    min_delta=0.0,
    patience=50,
    mode='min',
    verbose=1,
    restore_best_weights=True,
    start_from_epoch=100
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=128,
    callbacks=[early_stopping],
    validation_split=0.1,
)

eval = model.evaluate(X_test, y_test)
print(f"Model evaluation {eval}")

# Convert X_test to a TensorFlow tensor
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)

# y_pred = model.predict(X_test)
y_pred = model.predict(X_test_tensor)

# Plot the original sine function and the predicted values
plt.figure(figsize=(16, 12))
# plot scatter, use small dot
plt.scatter(X_test[:, 0], y_test, label='Test Data', marker='x', s=3)
# plt.scatter(X_train[:, 0], y_train, label='Train Data', marker='x', s=1)
# sort X_test by first column to plot line
# X_test = X_test[X_test[:, 0].argsort()]
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted Values', marker='o', s=3)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Function Approximation using TensorFlow')
plt.savefig('function_approximation.png')
plt.show()
