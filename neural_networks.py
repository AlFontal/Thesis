#!/usr/bin/env python

import tensorflow as tf
import numpy as np

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