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
    return tf.Variable(initial)

def bias_variable(shape, name="B"):
    """Provides a tensor of bias variables with value 0.1"""


    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



input_tensor = preprocess.train_tensor  # Import train set

test_set = preprocess.test_tensor

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
x = tf.placeholder(tf.float32, [None, seq_len * aa_vec_len], name="x")
y_ = tf.placeholder(tf.float32, [None, n_labels], name="labels")

y = fc_layer(x, seq_len * aa_vec_len, n_labels, relu=False, name="fc")

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
writer = tf.summary.FileWriter("/home/alejandro/Documents/GitHub/Thesis/logs")
writer.add_graph(sess.graph)

for i in range(n_iters):
    i += 1
    a, b = get_batch(input_tensor, n=minibatch_size)
    train_step.run(feed_dict={x: a, y_: b})
    tf.summary.scalar("Cross Entropy", cross_entropy)

    """
    if i % 5 == 0:
        [train_accuracy, s] = sess.run([accuracy, summ],
                                       feed_dict={x: a, y_: b})
        writer.add_summary(s, i)
    """

    if i % 100 == 0 or i == 1:  # Check only in iterations multiple of 100
        a, b = get_batch(input_tensor, n=len(input_tensor))  # Check in full train set
        train_acc = accuracy.eval(feed_dict={x: a, y_: b})
        test_acc = accuracy.eval(feed_dict={x: x_test, y_: y_test})
        print "Iteration number " + str(i) + ":\n"
        print "Train accuracy: {}%\t Test Accuracy: {}%\n".format(
            round(train_acc*100, 3), round(test_acc*100, 2))

        if train_acc == 1:
            break


