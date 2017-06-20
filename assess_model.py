from preprocess import *
from neural_networks import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sys import argv
import seaborn as sns

__author__ = 'fonta004'

curr_dir = os.getcwd()
seqdir = curr_dir + "/seqs/" + str(argv[1]) + ".fasta"

n_units_1 = 100
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
    for i in range(len(seq) - w + 1):
        mod_seq = seq[:i] + "X" * w + seq[i+w:]
        mod_seqs.append(mod_seq)

    return [seq] + mod_seqs


test_seq = "MGAGSSTEQRSPEQPPEGSSTPAEPEPSGGGPSAEAAPDTTADPAIAASDPATKLLQKNGQLST" \
           "INGVAEQDELSLQEGDLNGQKGALNGQGALNSQEEEEVIVTEVGQRDSEDVSERDSDKEMATKS" \
           "AVVHDITDDGQEENRNIEQIPSSESNLEELTQPTESQANDIGFKKVFKFVGFKFTVKKDKTEKP" \
           "DTVQLLTVKKDEGEGAAGAGDHQDPSLGAGEAASKESEPKQSTEKPEETLKREQSHAEISPPAE" \
           "SGQAVEECKEEGEEKQEKEPSKSAESPTSPVTSETGSTFKKFFTQGWAGWRKKTSFRKPKEDEV" \
           "EASEKKKEQEPEKVDTEEDGKAEVASEKLTASEQAHPQEPAESAHEPRLSAEYEKVELPSEEQV" \
           "SGSQGPSEEKPAPLATEVFDEKIEVHQEEVVAEVHVSTVEERTEEQKTEVEETAGSVPAEELVG" \
           "MDAEPQEAEPAKELVKLKETCVSGEDPTQGADLSPDEKVLSKPPEGVVSEVEMLSSQERMKVQG" \
           "SPLKKLFTSTGLKKLSGKKQKGKRGGGDEESGEHTQVPADSPDSQEEQKGESSASSPEEPEEIT" \
           "CLEKGLAEVQQDGEAEEGATSDGEKKREGVTPWASFKKMVTPKKRVRRPSESDKEDELDKVKSA" \
           "TLSSTESTASEMQEEMKGSVEEPKPEEPKRKVDTSVSWEALICVGSSKKRARRRSSSDEEGGPK" \
           "AMGGDHQKADEAGKDKETGTDGILAGSQEHDPGQGSSSPEQAGSPTEGEGVSTWESFKRLVTPR" \
           "KKSKSKLEEKSEDSIAGSGVEHSTPDTEPGKEESWVSIKKFIPGRRKKRPDGKQEQAPVEDAGP" \
           "TGANEDDSDVPAVVPLSEYDAVEREKMEAQQAQKGAEQPEQKAATEVSKELSESQVHMMAAAVA" \
           "DGTRAATIIEERSPSWISASVTEPLEQVEAEAALLTEEVLEREVIAEEEPPTVTEPLPENREAR" \
           "GDTVVSEAELTPEAVTAAETAGPLGSEEGTEASAAEETTEMVSAVSQLTDSPDTTEEATPVQEV" \
           "EGGVPDIEEQERRTQEVLQAVAEKVKEESQLPGTGGPEDVLQPVQRAEAERPEEQAEASGLKKE" \
           "TDVVLKVDAQEAKTEPFTQGKVVGQTTPESFEKAPQVTESIESSELVTTCQAETLAGVKSQEMV" \
           "MEQAIPPDSVETPTDSETDGSTPVADFDAPGTTQKDEIVEIHEENEVASGTQSGGTEAEAVPAQ" \
           "KERPPAPSSFVFQEETKEQSKMEDTLEHTDKEVSVETVSILSKTEGTQEADQYADEKTKDVPFF" \
           "EGLEGSIDTGITVSREKVTEVALKGEGTEEAECKKDDALELQSHAKSPPSPVEREMVVQVEREK" \
           "TEAEPTHVNEEKLEHETAVTVSEEVSKQLLQTVNVPIIDGAKEVSSLEGSPPPCLGQEEAVCTK" \
           "IQVQSSEASFTLTAAAEEEKVLGETANILETGETLEPAGAHLVLEEKSSEKNEDFAAHPGEDAV" \
           "PTGPDCQAKSTPVIVSATTKKGLSSDLEGEKTTSLKWKSDEVDEQVACQEVKVSVAIEDLEPEN" \
           "GILELETKSSKLVQNIIQTAVDQFVRTEETATEMLTSELQTQAHVIKADSQDAGQETEKEGEEP" \
           "QASAQDETPITSAKEESESTAVGQAHSDISKDMSEASEKTMTVEVEGSTVNDQQLEEVVLPSEE" \
           "EGGGAGTKSVPEDDGHALLAERIEKSLVEPKEDEKGDDVDDPENQNSALADTDASGGLTKESPD" \
           "TNGPKQKEKEDAQEVELQEGKVHSESDKAITPQAQEELQKQERESAKSELTES"


test_seq2 = "MTAVSKAFEFLDDDRIRVTTSDNTQGFYLKEVLMKPKQAVFLQFDQQYKSSTATLFGAMV" \
            "AHTMRILQANQNREAVFIALDEIINCAPIPKFTDLLNTIRSANMPTFLYLQSLEGLNRLY" \
            "GANSDKMFMGSSNLKIVFRIGDIESAEECSRLVGQTETTYISETAGTSQTSGTSSSSRAS" \
            "SSSSNRSQNTGTTKSIKLESIIEPAEFIKLPICTAVVMYNGSYGTLEMPKYYECYNMPKR" \
            "TNLKTIRDFKVA"


test_seq3 = "MTAVSKAFEFLDDDRIRVTTSDNTQGFYLKEVLMKPKQAVFLQFDQQYKSSTATLFGAMV" \
            "MEQAIPPDSVETPTDSETDGSTPVADFDAPGTTQKDEIVEIHEENEVASGTQSGGTEAEAVPAQ" \
            "KERPPAPSSFVFQEETKEQSKMEDTLEHTDKEVSVETVSILSKTEGTQEADQYADEKTKDVPFF" \
            "EGLEGSIDTGITVSREKVTEVALKGEGTEEAECKKDDALELQSHAKSPPSPVEREMVVQVEREK" \
            "TEAEPTHVNEEKLEHETAVTVSEEVSKQLLQTVNVPIIDGAKEVSSLEGSPPPCLGQEEAVCTK" \
            "IQVQSSEASFTLTAAAEEEKVLGETANILETGETLEPAGAHLVLEEKSSEKNEDFAAHPGEDAV" \
            "PTGPDCQAKSTPVIVSATTKKGLSSDLEGEKTTSLKWKSDEVDEQVACQEVKVSVAIEDLEPEN" \
            "GILELETKSSKLVQNIIQTAVDQFVRTEETATEMLTSELQTQAHVIKADSQDAGQETEKEGEEP" \
            "QASAQDETPITSAKEESESTAVGQAHSDISKDMSEASEKTMTVEVEGSTVNDQQLEEVVLPSEE" \
            "EGGGAGTKSVPEDDGHALLAERIEKSLVEPKEDEKGDDVDDPENQNSALADTDASGGLTKESPD" \
            "TNGPKQKEKEDAQEVELQEGKVHSESDKAITPQAQEELQKQERESAKSELTES"


seq_len = 500

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, seq_len * 27], name="x")
    y_ = tf.placeholder(tf.float32, [None, 11], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="dropout_rate")


fc1 = fc_layer(x, seq_len * 27, n_units_1, relu=False, name="fc1")
fc1_drop = tf.nn.dropout(fc1, keep_prob)
y = fc_layer(fc1_drop, n_units_1, 11, relu=False, name="fc2")

saver = tf.train.Saver()

sess = tf.InteractiveSession()  # Start tensorflow session

# Restore variables from disk (Specify network location)
saver.restore(sess, "logs/2017-06-16 12:54:32-0.1-500-drop0.6-fc_2l(100x11)"
                    "(seq+props)seqlen=500/model.ckpt")


raw_seqs = fasta_parse(seqdir)

# raw_seqs = [test_seq, test_seq2, test_seq3]

seqs = seq_process(raw_seqs, seq_len)

prog_decile = round(len(seqs)/10)

for n, curr_seq in enumerate(seqs):
    if n % prog_decile == 0:
        print "Progress = {0} %".format(str(n * 10 / prog_decile))

    curr_seq = seq_process([curr_seq], seq_len)[0]
    test_seqs5 = break_seq(curr_seq, 5)
    test_seqs15 = break_seq(curr_seq, 15)

    t_tens5 = []
    for seq in test_seqs5:
        t_tens5.append(sum(seq2onehot(seq, aa_dict), []))

    t_tens15 = []
    for seq in test_seqs15:
        t_tens15.append(sum(seq2onehot(seq, aa_dict), []))

    lab = [np.zeros(11)]  # Won't be used, but needed to feed value

    # Extract output layer of original seq and apply softmax.
    # Divided by 1000 in order to make softmax results somehow variable.

    original_out = np.divide(y.eval(
        feed_dict={x: [t_tens5[0]], y_: lab, keep_prob: 1}), 1000)
    final_label = tf.nn.softmax(original_out).eval()

    # Redefine crossentropy, now taking as label the output of the original seq
    with tf.name_scope("crossentropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=final_label,
                                                logits=np.divide(y, 1000)))
        tf.summary.scalar("crossentropy", cross_entropy)

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


mean5 = cross5s.mean(axis=0)
mean15 = cross15s.mean(axis=0)

pos1 = np.arange(1, len(mean5)+1)
pos2 = np.arange(1, len(mean15)+1)

plt.plot(pos1, mean5, "darkblue", pos2, mean15, "darkred", linewidth=2.0)

plt.xlabel("Position in sequence")
plt.ylabel("Increase in CrossEntropy")
plt.title("CrossEntropy perturbation per position in {} sequences".format(
    str(argv[1])))

blue_patch = mpatches.Patch(color='darkblue', label='window = 15 aa')
red_patch = mpatches.Patch(color='darkred', label='window = 5 aa')
plt.legend(handles=[blue_patch, red_patch], loc=1)

plt.savefig("{}.png".format(str(argv[1])))