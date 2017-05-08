#!/usr/bin/env python

from __future__ import division
import numpy as np
import tensorflow as tf
import preprocess

n_labels = 10
aa_vec_len = 21
seq_len = 1000
n_iters = 2000
minibatch_size = 500
learn_step = 0.5


def get_batch(tensor, n=100):
    """
    Takes a tensor of shape t = [[[seq_1][lab_1]], ..., [[seq_n][lab_n]]] and
    randomly takes n samples, returning a tensor x = [[seq_1], ..., [seq_n]]
    and a tensor y = [[lab_1], ..., [lab_n]].

    """
    idxs = np.random.choice(len(tensor), n, replace=False)
    x = [tensor[i][0] for i in idxs]
    y = [tensor[i][1] for i in idxs]

    return x, y


def fc_layer(input, channels_in, channels_out, name="fc", relu=True):
    """
    Computes a fully connected layer when provided with an input tensor and
    returns an output tensor. Input and output channels must be specified.
    By default, the output uses a ReLu activation function.

    """

    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out],
                                            stddev=0.1), name="W")
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        out = tf.matmul(input, w) + b

        if relu:
            return tf.nn.relu(out)
        else:
            return out

input = preprocess.train_tensor  # Import train set

test_set = preprocess.test_tensor

x_test = [test_set[i][0] for i in range(len(test_set))]
y_test = [test_set[i][1] for i in range(len(test_set))]

sess = tf.InteractiveSession()  # Start tensorflow session

# Define variables of the network:
x = tf.placeholder(tf.float32, [None, seq_len * aa_vec_len])
y_ = tf.placeholder(tf.float32, [None, n_labels])
b = tf.Variable(tf.zeros([n_labels]))
W = get_weights([seq_len * aa_vec_len, n_labels])
y = tf.matmul(x, W) + b

tf.global_variables_initializer().run()  # Initialize variables

#  Define cost_function (cross entropy):
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(learn_step).\
    minimize(cross_entropy)



n = 0  # Set counter for number of iterations

for _ in range(n_iters):
    n += 1
    a, b = get_batch(input, n=minibatch_size)
    train_step.run(feed_dict={x: a, y_: b})
    if n % 100 == 0 or n == 1:  # Check only in iterations multiple of 100
        a, b = get_batch(input, n=len(input))  # Check in full train set
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_acc = accuracy.eval(feed_dict={x: a, y_: b})
        test_acc = accuracy.eval(feed_dict={x: x_test, y_: y_test})
        print "Iteration number " + str(n) + ":\n"
        print "Train accuracy: {}%\t Test Accuracy: {}%\n".format(
            round(train_acc*100, 3), round(test_acc*100, 2))

        if train_acc == 1:
            break


