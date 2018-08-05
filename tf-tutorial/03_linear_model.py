"""
    Created by nguyenvanhieu.vn on 05/08/2018
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1: Make sample notMnist.
X_train = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
Y_train = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = X_train.shape[0]

# Step 2: create placeholders for X_train and Y_train
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: build model to predict Y
Y_predicted = w * X + b

# Step 5: use the squared error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    # Step 8: train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in zip(X_train, Y_train):
            # Session execute optimizer and fetch values of loss
            _, _loss = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += _loss
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])
    # Y_pred = sess.run(Y_predicted, feed_dict={X: X_train, Y: Y_train}) # Can also get Y_pred with sess.run

print('Took: %f seconds' % (time.time() - start))

# Get predicted after train
Y_pred = X_train * w_out + b_out

# See diff between real and predict value
for i, j in zip(Y_pred, Y_train):
    print(i, '|', j)

# plot the results
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X_train, Y_train, 'bo', label='Real data')
plt.plot(X_train, Y_pred, 'r', label='Predicted')
plt.legend()
plt.show()
