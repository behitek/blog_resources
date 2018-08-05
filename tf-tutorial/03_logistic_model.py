"""
    Created by nguyenvanhieu.vn at 8/5/2018
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30

# Step 1: Read notMnist data
X_train, X_validation, X_test = utils.read_mnist('notMnist')
X_batch, Y_batch = utils.next_batch(batch_size, X_train)

# Step 2: create placeholders for features and labels
# each image in the notMnist is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to char A - J.
# each lable is one hot vector.
X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
Y = tf.placeholder(tf.int32, [batch_size, 10], name='label')

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf-tutorial.matmul(X, w)
# shape of b depends on Y
w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = tf.matmul(X, w) + b

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)  # computes the mean over all the examples in the batch
# loss = tf-tutorial.reduce_mean(-tf-tutorial.reduce_sum(tf-tutorial.nn.softmax(logits) * tf-tutorial.log(Y), reduction_indices=[1]))

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(len(X_train[0]) / batch_size)

    # train the model n_epochs times
    for i in range(n_epochs):
        total_loss = 0

        for j in range(n_batches):
            X_batch, Y_batch = utils.next_batch(batch_size, X_train)
            _, loss_batch = sess.run([optimizer, loss], {X: X_batch, Y: Y_batch})
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    n_batches = int(len(X_test[0]) / batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = utils.next_batch(batch_size, X_test)
        accuracy_batch = sess.run(accuracy, {X: X_batch, Y: Y_batch})
        total_correct_preds += accuracy_batch

    print('Accuracy {0}'.format(total_correct_preds / len(X_test[0]) * 100))

writer.close()