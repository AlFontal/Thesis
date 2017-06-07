__author__ = 'fonta004'

from preprocess import *
from neural_networks import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

props_file = "aa_propierties.csv"
aa_string = "ARNDCEQGHILKMFPSTWYVX"
aa_dict = get_1h_dict(aa_string, props_file, add_props=True)

def break_seq(seq, w=5):
    """
    Modifies a sequence replacing aa in a certain window w by Xs

    :param seq: String containing the sequence to break/modify
    :param w: Size of the window. Default of 5
    :return: List of strings, each string being a modified version of the seq.
    """

    mod_seqs = []
    for i in range(len(seq) - w +1):
        mod_seq = seq[:i] + "X" * w + seq[i+w:]
        mod_seqs.append(mod_seq)

    return [seq] + mod_seqs


test_seq = "TMDKSELVQKAKLAEQAERYDDMAAAMKAVTEQGHELSNEERNLLSVAYKNVVGARRSSWRVI" \
           "SSIEQKTERNEKKQQMGKEYREKIEAELQDICNDVLELLDKYLIPNATQPESKVFYLKMKGDY" \
           "FRYLSEVASGDNKQTTVSNSQQAYQEAFEISKKEMQPTHPIRLGLALNFSVFYYEILNSPEKA" \
           "CSLAKTAFDEAIAELDTLNEESYKDSTLIMQLLRDNLTLWTSENQGDEGDAGEGEN"


seq_len = 100

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, seq_len * 27], name="x")
    y_ = tf.placeholder(tf.float32, [None, 11], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="dropout_rate")


fc1 = fc_layer(x, seq_len * 27, seq_len, relu=False, name="fc1")
fc1_drop = tf.nn.dropout(fc1, keep_prob)
y = fc_layer(fc1_drop, seq_len, 11, relu=False, name="fc2")


saver = tf.train.Saver()

sess = tf.InteractiveSession()  # Start tensorflow session

# Restore variables from disk (Specify network location)
saver.restore(sess, "logs/2017-06-06 12:23:55-0.1-500-drop0.6-fc_2l(100x11)"
                    "(seq+props)seqlen=100/model.ckpt")

test_seq = seq_process([test_seq], seq_len)[0]
test_seqs5 = break_seq(test_seq, 5)
test_seqs15 = break_seq(test_seq, 15)

t_tens5 = []
for seq in test_seqs5:
    t_tens5.append(sum(seq2onehot(seq, aa_dict), []))

t_tens15 = []
for seq in test_seqs15:
    t_tens15.append(sum(seq2onehot(seq, aa_dict), []))


lab = [np.zeros(11)]  # Won't be used, but needed to feed value to placeholder.

# Extract output layer of original seq and apply softmax.
# Divided by 1000 in order to make softmax results somehow variable.

original_out = np.divide(y.eval(
    feed_dict={x: [t_tens5[0]], y_: lab, keep_prob: 1}), 1000)
final_label = tf.nn.softmax(original_out).eval()

# Redefine crossentropy, now taking as label the output of the original seq.
with tf.name_scope("crossentropy"):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=final_label,
                                                logits=np.divide(y, 1000)))
    tf.summary.scalar("crossentropy", cross_entropy)

# Calculate crossentropy of original sequence
xent = cross_entropy.eval(
    feed_dict={x: [t_tens5[0]], y_: final_label, keep_prob: 1})

cross5 = []
for seq in t_tens5[1:]:  # For each of the modified sequences
    # Store the difference in crossentropy respect the original sequence
    cross5.append(cross_entropy.eval(
        feed_dict={x: [seq], y_: lab, keep_prob: 1}) - xent)

cross15 = []
for seq in t_tens15[1:]:  # For each of the modified sequences
    # Store the difference in crossentropy respect the original sequence
    cross15.append(cross_entropy.eval(
        feed_dict={x: [seq], y_: lab, keep_prob: 1}) - xent)

pos1 = np.arange(1, len(cross5)+1)
pos2 = np.arange(1, len(cross15)+1)
plt.plot(pos1, cross5, "darkred", pos2, cross15, "darkgreen", linewidth=2.0)
plt.xlabel("Position in sequence")
plt.ylabel("Increase in CrossEntropy")
plt.title('CrossEntropy perturbation per position')
plt.grid()
green_patch = mpatches.Patch(color='darkgreen', label='window = 15 aa')
red_patch = mpatches.Patch(color='darkred', label='window = 5 aa')
plt.legend(handles=[green_patch, red_patch], loc= 1)

plt.show()
