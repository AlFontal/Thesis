#!/usr/bin/env python

from __future__ import division
from neural_networks import *
import numpy as np
import tensorflow as tf
import os

curr_dir = os.getcwd()
model_dir ="logs2/2017-10-31 13:39:35-0.02-500-drop0.8-100x11)(seq+props)" \
           "seqlen=750forwardLSTM"
seq_len = 750
aa_vec_len = 20
n_labels = 11
n_units_lstm = 100
n_units_fc = 250
n_timesteps = seq_len
out_lstm_size = 100
learn_step = 0.1
train_steps = 1000


# Redefine input to contain optimizable variables instead of placeholders

with tf.name_scope("input"):
    # optimal_x is initialized as a vector of seq_len x [0.05, 0.05, ..., 0.05]
    optimal_x = trainable_input([1, seq_len, aa_vec_len])
    tf.add_to_collection("class_optimizing", optimal_x)
    x_back = tf.reverse(optimal_x, [1])
    y_ = tf.placeholder(tf.float32, [None, n_labels], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="dropout_rate")

# Define graph structure to make it callable
pre_lstm = tf.unstack(x_back, n_timesteps, 1)
post_lstm = LSTM(pre_lstm, n_units_lstm, out_lstm_size, name="backward")
fc1 = tf.nn.dropout(
    fc_layer(post_lstm, out_lstm_size, n_units_fc, relu=True, name="fc1"),
    keep_prob)
y = fc_layer(fc1, n_units_fc, n_labels, relu=True, name="fc2")


# Define score function, score (S) is the unscaled logit for the label
with tf.name_scope("class_score"):
    # y is the last layer, contains unscaled logits
    # y_ is a one-hot vector containing the true label
    pos = tf.argmax(y_, 1)
    score = -tf.gather(y[0], pos) # Negative so that it is minimized
    tf.summary.scalar("class_score", score)

# Define train step to optimize only the input variables
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learn_step).minimize(
        score, var_list=tf.get_collection("class_optimizing"))


# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()
sess = tf.InteractiveSession()  # Start tensorflow session
sess.run(init_op)

# Restore variables from disk (Specify network location)
saver = tf.train.import_meta_graph(model_dir + "/model.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint(model_dir + '/./'))

label_tensor = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]


def standardize_tensor(tensor):
    """
    Takes a tensor and performs a row-wise standardization, returning a tensor
    of the same shape.
    """
    sums = tf.reduce_sum(tensor, 2)
    sums = tf.expand_dims(sums, 2)
    final_tensor = tensor / sums
    return final_tensor


for i in range(train_steps):
    train_step.run(feed_dict={y_: label_tensor, keep_prob: 1})
    # make negative values = 0
    no_neg_x = tf.where(tf.less(optimal_x, tf.zeros_like(optimal_x)),
                        tf.zeros_like(optimal_x), optimal_x)
    # standardize all values to sum 1 in a row per row basis
    optimal_x = standardize_tensor(no_neg_x)

    t, opt_x = sess.run([score, optimal_x],
                        feed_dict={y_: label_tensor, keep_prob: 1})

    opt_x = opt_x[0]
    print "Step {} \t Score = {}".format(str(i), str(-t))

# Save dataframe containing the input values as a text file
pssm_df = pd.DataFrame(opt_x)
pssm_df.columns = list("ARNDCEQGHILKMFPSTWYV")

with open(os.path.join(curr_dir, "sequence_pssm.txt"), 'w') as outfile:
    pssm_df.to_string(outfile)