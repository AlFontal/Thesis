#!/usr/bin/env python

from __future__ import division
import numpy as np
import tensorflow as tf
import os
import preprocess
from neural_networks import *
from time import gmtime, strftime
from sys import argv
import pandas as pd

datetime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
curr_dir = os.getcwd()

save_model = True
save_filters = True
print_progress = True
add_props = False

seqdir = curr_dir + "/seqs/"
seqfiles = os.listdir(seqdir)
props_file = "aa_propierties.csv"

seq_len = int(argv[3])
dataset = preprocess.DataSet(seqdir, props_file, add_props, seq_len,
                             flatten=True)

test_dict = dataset.test_dict
input_tensor = dataset.train_tensor  # Import train set
test_set = dataset.test_tensor
labels = dataset.labels

n_labels = len(labels)
aa_vec_len = len(dataset.aa_dict.values()[0])
n_epochs = 1000
minibatch_size = 500
learn_step = 0.1
iters_x_epoch = int(round(len(input_tensor)/minibatch_size, 0))
drop_prob = float(argv[1])
n_units_1 = int(argv[2])
n_filters = 5
maxpool_width = 5
pool_len = int(seq_len/maxpool_width)
n_final_features = pool_len * aa_vec_len * n_filters * 3

# Create logs directory for visualization in TensorBoard
logdir = "/logs2/{}-{}-{}-drop{}-conv-fc{}x11)(seq)seqlen={}LEAKY".format(
    datetime, learn_step, minibatch_size, drop_prob, n_units_1, seq_len)

os.makedirs(curr_dir + logdir + "/train")
os.makedirs(curr_dir + logdir + "/test")

x_test = [test_set[i][0] for i in range(len(test_set))]
y_test = [test_set[i][1] for i in range(len(test_set))]

sess = tf.InteractiveSession()  # Start tensorflow session

# Define variables of the network:

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, seq_len, aa_vec_len], name="x")
    x1 = tf.expand_dims(x, -1)
    y_ = tf.placeholder(tf.float32, [None, n_labels], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="dropout_rate")

# 3 convolutional layers, with filters of 5, 10 and 15 aminoacids
conv5 = conv_layer(x1, 5, aa_vec_len, 1, n_filters, name="conv5")
conv10 = conv_layer(x1, 10, aa_vec_len, 1, n_filters, name="conv10")
conv15 = conv_layer(x1, 15, aa_vec_len, 1, n_filters, name="conv15")

# Concatenate the outputs so we end up with a tensor of seqlen x 26 x 75
conv_conc = tf.concat([conv5, conv10, conv15], 3)

# Max Pooling layer. Reduce dimensionality and extract only significant signals
maxpooled = max_pool_layer(conv_conc, maxpool_width, aa_vec_len, n_filters * 3)

# Reshape to make input acceptable for fc layer
conv_out = tf.reshape(maxpooled, [-1, n_final_features])

fc1 = tf.nn.dropout(fc_layer(conv_out, n_final_features,
               n_units_1, relu=True, name="fc1"), keep_prob)

y = fc_layer(fc1, n_units_1, n_labels, relu=True, name="fc2")


# Define cost_function (cross entropy):
with tf.name_scope("crossentropy"):
    # y is the last layer, contains unscaled logits
    # y_ is a one-hot vector containing the true label
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    # Unscaled logits are normalized and crossentropy is calculated
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

# Save variable names of convolutional filters for later storage.
filts5 = [v for v in tf.trainable_variables() if v.name == "conv5/weights:0"]
bias5 = [v for v in tf.trainable_variables() if v.name == "conv5/B:0"]
filts10 = [v for v in tf.trainable_variables() if v.name == "conv10/weights:0"]
bias10 = [v for v in tf.trainable_variables() if v.name == "conv10/B:0"]
filts15 = [v for v in tf.trainable_variables() if v.name == "conv15/weights:0"]
bias15 = [v for v in tf.trainable_variables() if v.name == "conv15/B:0"]
filters = [filts5, filts10, filts15]
biases = [bias5, bias10, bias15]

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

a, b = get_batch(input_tensor, n=1)
or_conv = conv15.eval(feed_dict={x: a, y_: b, keep_prob: 1})
resh_conv = conv_out.eval(feed_dict={x: a, y_: b, keep_prob: 1})


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
        train_acc = accuracy.eval(
            feed_dict={x: a, y_: b, keep_prob: 1})
        test_acc = accuracy.eval(
            feed_dict={x: x_test, y_: y_test, keep_prob: 1})

        xent = cross_entropy.eval(
            feed_dict={x: a, y_: b, keep_prob: 1})

        _, t = sess.run([accuracy, summ],
                        feed_dict={x: x_test, y_: y_test, keep_prob: 1})

        # Generate confusion matrix and make it a Pandas dataframe
        cm = conf_mat.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1})
        df = pd.DataFrame(cm)
        df.columns = labels
        df.index = labels

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            best_train_acc = train_acc

            if save_filters and test_acc > 0.6:  # Store conv filters
                best_filters = []
                for filt in filters:
                    best_filters.append(sess.run(filt[0]))

            if save_model and test_acc > 0.6:
                # Save the variables to disk.
                save_path = saver.save(sess, "." + logdir + "/model.ckpt")
                print("Model saved!")
                df.to_csv(curr_dir + logdir + "/conf_matrix.csv")

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

        if train_acc > 0.98:
            break

if save_filters:
    pssms = []
    for filt in best_filters:
        pssms.append(preprocess.get_pssm_dict(filt, n_filters,
                                              dataset.aa_string))

    for i, pssm_dict in enumerate(pssms):
        filt_dir = curr_dir + logdir + "/filter" + str((i + 1) * 5)
        os.makedirs(filt_dir)
        os.chdir(filt_dir)
        for fn, pssm_mat in enumerate(pssm_dict.values()):
            pssm_mat = pssm_mat + biases[i][fn]
            base_filename = filt_dir +'/filter_{}.txt'.format(fn + 1)
            with open(os.path.join(curr_dir, base_filename), 'w') as outfile:
                pssm_mat.to_string(outfile)

            preprocess.run_seq2logo(base_filename, "filter" + str(fn + 1))

print str(round(max_test_acc * 100, 2))



