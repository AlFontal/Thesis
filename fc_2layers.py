#!/usr/bin/env python

from __future__ import division
import numpy as np
import tensorflow as tf
import os
import preprocess
from time import gmtime, strftime

datetime = strftime("%Y-%m-%d %H:%M:%S", gmtime())

curr_dir = os.getcwd()

input_tensor = preprocess.train_tensor  # Import train set

test_set = preprocess.test_tensor

trainset_size = len(input_tensor)
n_labels = 11
aa_vec_len = 21
seq_len = 1000
n_epochs = 4000
minibatch_size = 250
learn_step = 0.1
iters_x_epoch = int(round(trainset_size/minibatch_size, 0))
drop_prob = 1

# Create logs directory for visualization in TensorBoard
logdir = "/logs/{}-{}-{}-drop{}-fc_2l".format(datetime, learn_step,
                                          minibatch_size, drop_prob)
os.makedirs(curr_dir + logdir + "/train")
os.makedirs(curr_dir + logdir + "/test")

def get_batch(tensor, n=100):
    """Gets a minibatch from a tensor

    Takes a tensor of shape t = [[[seq_1][lab_1]], ..., [[seq_n][lab_n]]] and
    randomly takes n samples, returning a tensor x = [[seq_1], ..., [seq_n]]
    and a tensor y = [[lab_1], ..., [lab_n]].
    """
    idxs = np.random.choice(len(tensor), n, replace=False)
    x = [tensor[i][0] for i in idxs]
    y = [tensor[i][1] for i in idxs]

    return x, y


def weight_variable(shape, name="W"):
    """Generates weight variables

    Provides a tensor of weight variables obtained from a truncated normal
    distribution with mean=0 and std=0.1. All values in range [-0.1, 0.1]
    """

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name="B"):
    """Provides a tensor of bias variables with value 0.1"""

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)



x_test = [test_set[i][0] for i in range(len(test_set))]
y_test = [test_set[i][1] for i in range(len(test_set))]

sess = tf.InteractiveSession()  # Start tensorflow session


def fc_layer(input_tensor, input_dim, output_dim, name="fc", relu=True):
    """Generates a fully connected layer with biases and weights

    Computes a fully connected layer when provided with an input tensor and
    returns an output tensor. Input and output channels must be specified.
    By default, the output uses a ReLu activation function.
    """

    with tf.name_scope(name):
        w = weight_variable([input_dim, output_dim])
        b = bias_variable([output_dim])
        out = tf.matmul(input_tensor, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)

        if relu:
            return tf.nn.relu(out)
        else:
            return out


# Define variables of the network:

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, seq_len * aa_vec_len], name="x")
    y_ = tf.placeholder(tf.float32, [None, n_labels], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="dropout_rate")


fc1 = fc_layer(x, seq_len * aa_vec_len, n_labels, relu=True, name="fc1")
fc1_drop = tf.nn.dropout(fc1, keep_prob)
y = fc_layer(fc1_drop, n_labels, n_labels, relu=False, name="fc2")

tf.global_variables_initializer().run()  # Initialize variables

#  Define cost_function (cross entropy):
with tf.name_scope("crossentropy"):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar("crossentropy", cross_entropy)


with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(learn_step).\
        minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(curr_dir + logdir + "/train")
train_writer.add_graph(sess.graph)
test_writer = tf.summary.FileWriter(curr_dir + logdir + "/test")
epoch_nr = 0

for i in range(n_epochs * iters_x_epoch):
    a, b = get_batch(input_tensor, n=minibatch_size)
    train_step.run(feed_dict={x: a, y_: b, keep_prob: drop_prob})

    _, s = sess.run([accuracy, summ],
                    feed_dict={x: a, y_: b, keep_prob: 1})
    train_writer.add_summary(s, i)

    if i % iters_x_epoch == 0:
        epoch_nr += 1
        # Check in full train set
        a, b = get_batch(input_tensor, n=len(input_tensor))
        train_acc = accuracy.eval\
            (feed_dict={x: a, y_: b, keep_prob: 1})
        test_acc = accuracy.eval\
            (feed_dict={x: x_test, y_: y_test, keep_prob: 1})
        xent = cross_entropy.eval\
            (feed_dict={x: a, y_: b, keep_prob: 1})

        _, t = sess.run([accuracy, summ],
                        feed_dict={x: x_test, y_: y_test, keep_prob: 1})

        test_writer.add_summary(t, i)
        print "Epoch number " + str(epoch_nr) + ":\n"
        print "Train accuracy: {}%\t Test Accuracy: {}%\t CrossEntropy: {}\n".\
            format(round(train_acc*100, 3), round(test_acc*100, 2), xent)

        if train_acc == 1:
            break


