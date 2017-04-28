#!/usr/bin/env python

from __future__ import division
import numpy as np
import tensorflow as tf
import preprocess

sess = tf.InteractiveSession()

seq1 = preprocess.seq1
seq2 = preprocess.seq2
test = [sum(preprocess.seq2onehot(seq2), []),
        sum(preprocess.seq2onehot(seq1), [])]


x = tf.placeholder(tf.float32, [None, 60*22])
y_ = tf.placeholder(tf.float32, [None, 2])
b = tf.Variable(tf.zeros([2]))
W = tf.Variable(tf.zeros([60*22, 2]))
y = tf.matmul(x, W) + b

tf.global_variables_initializer().run()
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


print cross_entropy.eval(feed_dict={x: test, y_: [[1 , 0], [0 , 1]]})


# Work in progress