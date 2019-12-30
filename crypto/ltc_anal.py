#!/usr/bin/env python3

from numpy import genfromtxt
import tensorflow as tf
import numpy as np
import random
import requests
import io
import matplotlib.pyplot as plt


def parseData(raw: np.ndarray, forward: int = 1):
    lg = len(raw)
    [velocity, acc, prev, current, amount, timeStamp, next] = [np.ndarray((lg, 1)) for i in range(7)]
    columns = [velocity, acc, prev, current, amount, timeStamp, next]
    ratios = [-1, -2, -4, -4, -1, -10, -4]

    data = np.ndarray((lg, len(columns)))

    for i in range(1, lg-forward):
        amount[i] = raw[i][2]
        prev[i] = raw[i-1][1]
        current[i] = raw[i][1]
        velocity[i] = current[i] - prev[i]
        acc[i] = velocity[i] - velocity[i-1]
        next[i] = raw[i+forward][1]
        timeStamp[i] = raw[i][0]

    data[:, 0] = amount[:, 0]

    for i in range(len(columns)):
        data[:, i] = columns[i][:, 0]
        data[:, i] *= 10**ratios[i]

    return data[:, 0:len(columns)-1], data[:, len(columns)-1]


def getDataset(entryData: np.ndarray, backSteps: int = 3, forwardSteps: int = 3):

    dataSet = np.ndarray((len(entryData)-forwardSteps, backSteps+1))
    for i in range(backSteps, len(entryData)-forwardSteps):
        for j in range(backSteps):
            dataSet[i][j] = btcEur[i-backSteps+j][1]
        dataSet[i][-1] = np.sum(btcEur[i:i+j, 2])

    labels = btcEur[forwardSteps:, 1]

    return dataSet, labels


my_data = genfromtxt('kraken.ltc.csv', delimiter=',')
btcEurRaw = requests.get("http://api.bitcoincharts.com/v1/trades.csv?symbol=krakenEUR")
btcEur = genfromtxt(io.StringIO(btcEurRaw.text), delimiter=",")

# dataSet, labels = getDataset(btcEur, 10, 20)
dataSet, labels = parseData(btcEur, forward=10)

trainSet = dataSet[2000:12000]
trainLabels = labels[2000:12000]

testSet = dataSet[-1010:-10]
testLabels = labels[-1010:-10]

model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(input_shape=(1000, 1)),
  tf.keras.layers.Dense(7, activation='tanh'),
  tf.keras.layers.Dense(29, activation='tanh'),
  tf.keras.layers.Dense(14, activation='tanh'),
  tf.keras.layers.Dense(3, activation='tanh'),
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

if input("Train? ") == "y":
    model.fit(trainSet, trainLabels, epochs=int(input("Epochs? ")))
    model.save_weights("lastFit.h5")
else:
    model.fit(trainSet, trainLabels, epochs=1)
    model.load_weights("bestFit.h5")

with open("lastModel.yaml", "w") as md:
    md.write(model.to_yaml())

model.evaluate(testSet, testLabels, verbose=1)
print(model.predict(testSet))
plt.plot(model.predict(testSet), label="prediction")
plt.plot(btcEur[-1010:-10, 1]/10000, label="reality")
plt.legend()
plt.title("Prediction to current BTC value")
plt.show()
