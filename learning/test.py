#!/usr/bin/env python3


import numpy as np


# sigmoid function
def nonlin(x, deriv=False):
    return x*(1-x) if deriv else 1/(1+np.exp(-x))


X = np.array([[1, 1],
              [3, 3],
              [-5, 2],
              [7, 1],
              [9, 3],
              ])
y = np.array([[1, 1, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((2,  1)) - 1

for _ in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)
