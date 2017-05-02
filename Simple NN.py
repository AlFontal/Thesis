#!/usr/bin/env python

from __future__ import division$
import numpy as np
import tensorflow as tf
import preprocess

n_labels = 10


def get_batch(tensor, n=100):

    idxs = np.random.choice(len(tensor), n, replace=False)
    x = [tensor[i][0] for i in idxs]
    y = [tensor[i][1] for i in idxs]

    return x, y

def get_weights(shape):
    w = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(w)


sess = tf.InteractiveSession()

input = preprocess.train_tensor

x = tf.placeholder(tf.float32, [None, 100*22])
y_ = tf.placeholder(tf.float32, [None, n_labels])
b = tf.Variable(tf.zeros([n_labels]))
W = get_weights([100*22, n_labels])
y = tf.matmul(x, W) + b

tf.global_variables_initializer().run()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

n = 0
test_set = preprocess.test_tensor
x_test = [test_set[i][0] for i in range(len(test_set))]
y_test = [test_set[i][1] for i in range(len(test_set))]

for _ in range(3000):
    n += 1
    a, b = get_batch(input, n=500)
    train_step.run(feed_dict={x: a, y_: b})
    if n % 100 == 0 or n == 1:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_acc = accuracy.eval(feed_dict={x: a, y_: b})
        test_acc = accuracy.eval(feed_dict={x: x_test, y_: y_test})
        print str(n) + "th iteration:"
        print "Train accuracy: {}%\t Test Accuracy: {}%\n".format(
            round(train_acc*100, 3), round(test_acc*100, 2))

