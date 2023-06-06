#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import numpy as np
import random
# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#  x_train, x_test = x_train / 25

x_train = np.random.random((1000, 1))/3
y_train = x_train * .2

xtst = np.random.random((10, 1))
ytst = xtst * .2

velkyArray = np.array(
    [
        [float(random.choice(range(48, 58))) for _ in range(10)]
        for _ in range(100)
    ]
)
mensiArray = velkyArray[:10]

velkyTrain = np.array([(sum(list(i)) - 48*10) / 100 for i in velkyArray])

mensiTrain = np.array([(sum(list(i)) - 48*10) / 100 for i in mensiArray])

model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(input_shape=(1000, 1)),
  tf.keras.layers.Dense(10, activation='tanh'),
  # tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='tanh'),
  tf.keras.layers.Dense(10, activation='linear'),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=1000)
# model.evaluate(xtst, ytst, verbose=1)
# vals = np.random.random((10, 1)) / 3
# p = model.predict(vals)
# for i in range(len(vals)):
#     print(f"v: {vals[i]}, r: {vals[i]*.2}, p: {p[i]}, (p-v)^2: {(p[i]-(vals[i]*.2))**2}")

# model.evaluate(x_test,  y_test, verbose=2)

model.fit(velkyArray, velkyTrain, epochs=1000)
model.evaluate(mensiArray, mensiTrain, verbose=1)
print(model.predict(mensiArray))
