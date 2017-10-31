__author__ = 'fonta004'

from preprocess import *
from neural_networks import *
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sys import argv
import seaborn as sns
n_filters = 25
aa_vec_len = 20
n_labels = 11
n_units_lstm = 100
n_units_fc = 250
seq_len = 750
n_timesteps = seq_len
model_dir ="logs2/2017-09-05 14:36:18-0.02-500-drop0.8-250x11)(seq+props)" \
           "seqlen=750biLSTM"
curr_dir = os.getcwd()
seqdir = curr_dir + "/seqs/" + str(argv[1]) + ".fasta"
props_file = "aa_propierties.csv"
aa_string = "ARNDCEQGHILKMFPSTWYV"
aa_dict = get_1h_dict(aa_string, props_file, add_props=False)


def break_seq(seq, w=5):
    """
    Modifies a sequence replacing aa in a certain window w by Xs

    :param seq: String containing the sequence to break/modify
    :param w: Size of the window. Default of 5
    :return: List of strings, each string being a modified version of the seq.
    """

    mod_seqs = []
    for i in range(len(seq) - w + 1):
        mod_seq = seq[:i] + "X" * w + seq[i+w:]
        mod_seqs.append(mod_seq)

    return [seq] + mod_seqs


# In order to recover variables, the graph structure must be defined previously

##############################################################################

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, seq_len, aa_vec_len], name="x")
    x_back = tf.reverse(x, [1])
    y_ = tf.placeholder(tf.float32, [None, n_labels], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="dropout_rate")


#  Unstack the tensor, 1 aa will be fed to each lstm cell each timepoint
pre_lstm1 = tf.unstack(x, n_timesteps, 1)
# pre_lstm2 = tf.unstack(x_back, n_timesteps, 1)

post_lstm1 = LSTM(pre_lstm1, n_units_lstm, 100, name="forward")
# post_lstm2 = LSTM(pre_lstm2, n_units_lstm, 100, name="backward")

# un_lstms = tf.concat([post_lstm1, post_lstm2], 1)
fc1 = tf.nn.dropout(fc_layer(post_lstm1, 100, n_units_fc,
                             relu=True, name="fc1"), keep_prob)

y = fc_layer(fc1, n_units_fc, n_labels, relu=True, name="fc2")

##############################################################################

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()
sess = tf.InteractiveSession()  # Start tensorflow session
sess.run(init_op)

# Restore variables from disk (Specify network location)
saver = tf.train.import_meta_graph(model_dir + "/model.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint(model_dir + '/./'))

# Parse and process sequences in fasta file in seqdir
raw_seqs = fasta_parse(seqdir)
seqs = seq_process(raw_seqs, seq_len)
lab = [np.zeros(11)]  # Won't be used, but needed to feed value
prog_decile = round(len(seqs)/10)

for n, curr_seq in enumerate(seqs):
    if n % prog_decile == 0:
        print "Progress = {0} %".format(str(n * 10 / prog_decile))
    # Break sequences in windows of 5 and 15 aa. Store the resulting tensors.
    test_seqs5 = break_seq(curr_seq, 5)
    test_seqs15 = break_seq(curr_seq, 15)
    t_tens5 = []
    for seq in test_seqs5:
        t_tens5.append(sum([seq2onehot(seq, aa_dict)], []))
    t_tens15 = []
    for seq in test_seqs15:
        t_tens15.append(sum([seq2onehot(seq, aa_dict)], []))

    # Extract output layer of original seq and apply softmax.
    pre_soft = y.eval(feed_dict={x: [t_tens5[0]], y_: lab, keep_prob: 1})
    # Scaled in order to make softmax results more sensitive to changes.
    maxval = tf.reduce_max(pre_soft).eval()
    scaled_pre_soft = np.divide(pre_soft, maxval * 2)
    # Save last layer of original sequence to compare with broken seqs
    final_label = tf.nn.softmax(scaled_pre_soft).eval()

    # Redefine crossentropy, now taking as label the output of the original seq
    with tf.name_scope("crossentropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=final_label,
                                        logits=np.divide(y, maxval*2)))

    # Calculate crossentropy of original sequence
    xent = cross_entropy.eval(
        feed_dict={x: [t_tens5[0]], y_: final_label, keep_prob: 1})

    cross5 = np.array([])
    for seq in t_tens5[1:]:  # For each of the modified sequences
        # Store the difference in crossentropy respect the original sequence
        cross5 = np.append(cross5, cross_entropy.eval(
            feed_dict={x: [seq], y_: lab, keep_prob: 1}) - xent)
        cross5 = np.array([cross5])

    cross15 = np.array([])
    for seq in t_tens15[1:]:  # For each of the modified sequences
        # Store the difference in crossentropy respect the original sequence
        cross15 = np.append(cross15, cross_entropy.eval(
            feed_dict={x: [seq], y_: lab, keep_prob: 1}) - xent)
        cross15 = np.array([cross15])

    if n == 0:
        cross5s = cross5
        cross15s = cross15

    else:
        cross5s = np.concatenate((cross5s, cross5), axis=0)
        cross15s = np.concatenate((cross15s, cross15), axis=0)

# Calculate per position mean difference of crossentropy
mean5 = cross5s.mean(axis=0)
mean15 = cross15s.mean(axis=0)
# Store values to .txt file
mean5.tofile(model_dir + "/mean5_" + argv[1] + ".txt", sep=" ")
mean15.tofile(model_dir + "/mean15_" + argv[1] + ".txt", sep=" ")

# Plot and save the results
pos1 = np.arange(1, len(mean5)+1)
pos2 = np.arange(1, len(mean15)+1)

sns.set_style("darkgrid")
plt.plot(pos1, mean5, "darkblue", pos2, mean15, "darkred", linewidth=2.0)

plt.xlabel("Position in sequence")
plt.ylabel("Increase in CrossEntropy")
plt.title("CrossEntropy perturbation per position in {} sequences".format(
    str(argv[1])))

blue_patch = mpatches.Patch(color='darkblue', label='window = 5 aa')
red_patch = mpatches.Patch(color='darkred', label='window = 15 aa')
plt.legend(handles=[blue_patch, red_patch], loc=1)

plt.savefig(model_dir + "/{}.png".format(str(argv[1])))