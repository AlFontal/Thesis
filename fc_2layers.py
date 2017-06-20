#!/usr/bin/env python

from __future__ import division
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import preprocess
from neural_networks import *
from time import gmtime, strftime
from sys import argv


datetime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
curr_dir = os.getcwd()

save_model = True
seqdir = curr_dir + "/seqs/"
seqfiles = os.listdir(seqdir)
props_file = "aa_propierties.csv"
add_props = True
seq_len = int(argv[3])
dataset = preprocess.DataSet(seqdir, props_file, add_props, seq_len)
test_dict = dataset.test_dict
input_tensor = dataset.train_tensor  # Import train set
test_set = dataset.test_tensor
labels = dataset.labels

trainset_size = len(input_tensor)
n_labels = len(labels)
aa_vec_len = len(dataset.aa_dict.values()[0])
n_epochs = 1000
minibatch_size = 500
learn_step = 0.1
iters_x_epoch = int(round(trainset_size/minibatch_size, 0))
drop_prob = float(argv[1])
n_units_1 = int(argv[2])
print_progress = False


# Create logs directory for visualization in TensorBoard
logdir = "/logs/{}-{}-{}-drop{}-fc_2l({}x11)(seq+props)seqlen={}".format(
    datetime, learn_step, minibatch_size, drop_prob, n_units_1, seq_len)

os.makedirs(curr_dir + logdir + "/train")
os.makedirs(curr_dir + logdir + "/test")

x_test = [test_set[i][0] for i in range(len(test_set))]
y_test = [test_set[i][1] for i in range(len(test_set))]

sess = tf.InteractiveSession()  # Start tensorflow session


# Define variables of the network:

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, seq_len * aa_vec_len], name="x")
    y_ = tf.placeholder(tf.float32, [None, n_labels], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="dropout_rate")


fc1 = fc_layer(x, seq_len * aa_vec_len, n_units_1, relu=False, name="fc1")
fc1_drop = tf.nn.dropout(fc1, keep_prob)
y = fc_layer(fc1_drop, n_units_1, n_labels, relu=False, name="fc2")


# Define cost_function (cross entropy):
with tf.name_scope("crossentropy"):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar("crossentropy", cross_entropy)


with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learn_step).\
        minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

with tf.name_scope("confusion_matrix"):
    lab = tf.argmax(y_, 1)
    pred = tf.argmax(y, 1)
    conf_mat = tf.contrib.metrics.confusion_matrix(
        labels=lab, predictions=pred)

tf.global_variables_initializer().run()  # Initialize variables

summ = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(curr_dir + logdir + "/train")
train_writer.add_graph(sess.graph)
test_writer = tf.summary.FileWriter(curr_dir + logdir + "/test")

epoch_nr = 0
max_test_acc = 0
best_train_acc = 0

class_dict = {}
writers_dict = {}
for label in labels:
    class_x = [test_dict[label][i][0] for i in range(len(test_dict))]
    class_y = [test_dict[label][i][1] for i in range(len(test_dict))]
    class_dict[label] = (class_x, class_y)
    writers_dict[label] = tf.summary.FileWriter(curr_dir + logdir
                                                          + "/" + label)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

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

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            best_train_acc = train_acc

        cm = conf_mat.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1})
        df = pd.DataFrame(cm)
        df.columns = labels
        df.index = labels

        test_writer.add_summary(t, i)

        for label in labels:
            class_x, class_y = class_dict[label]
            acc = accuracy.eval(feed_dict=
                                {x: class_x, y_: class_y, keep_prob: 1})

            _, c = sess.run([accuracy, summ],
                            feed_dict={x: class_x, y_: class_y, keep_prob: 1})

            writers_dict[label].add_summary(c, i)

        if print_progress:
            print "Epoch number " + str(epoch_nr) + ":\n"
            print "Train accuracy: {}%\t Test Accuracy: {}% \t" \
                  " CrossEntropy: {}\n".format(round(train_acc*100, 3),
                   round(test_acc*100, 2),  xent)

            print df

        if train_acc > 0.98:
            df.to_csv(curr_dir + logdir + "/conf_matrix.csv")
            break

print str(round(max_test_acc * 100, 2))


if save_model:
    # Save the variables to disk.
    save_path = saver.save(sess, "." + logdir + "/model.ckpt")
    print("Model saved in file: %s" % save_path)

