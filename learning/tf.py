#!/usr/bin/env python3

from __future__ import print_function

# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Define model
nrNodesHiddenLayer1 = 8 # 1st layer of neurons
nrNodesHiddenLayer2 = 8 # 2nd layer of neurons
nrNodesHiddenLayer3 = 4
nrClasses = 10
batchSize = 128

# Placeholders
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# Create neural network model
def neuralNetworkModel(data):
    hidden1Layer = {'weights':tf.Variable(tf.random_normal([784, nrNodesHiddenLayer1])),
                      'biases':tf.Variable(tf.random_normal([nrNodesHiddenLayer1]))}

    hidden2Layer = {'weights':tf.Variable(tf.random_normal([nrNodesHiddenLayer1, nrNodesHiddenLayer2])),
                      'biases':tf.Variable(tf.random_normal([nrNodesHiddenLayer2]))}

    hidden3Layer = {'weights':tf.Variable(tf.random_normal([nrNodesHiddenLayer2, nrNodesHiddenLayer3])),
                      'biases':tf.Variable(tf.random_normal([nrNodesHiddenLayer3]))}

    outputLayer = {'weights':tf.Variable(tf.random_normal([nrNodesHiddenLayer3, nrClasses])),
                    'biases':tf.Variable(tf.random_normal([nrClasses]))}

    # create flow
    l1 = tf.add(tf.matmul(data,hidden1Layer['weights']), hidden1Layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden2Layer['weights']), hidden2Layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden3Layer['weights']), hidden3Layer['biases'])
    l3 = tf.nn.relu(l3)

    return tf.matmul(l3,outputLayer['weights']) + outputLayer['biases']

def trainNeuralNetwork(x):
    prediction = neuralNetworkModel(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
    no_epochs = 100

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(no_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples/batchSize)):
                epoch_x, epoch_y = mnist.train.next_batch(batchSize)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

                print('Epoch', epoch, 'completed out of',no_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


trainNeuralNetwork(x)
