#!/usr/bin/env python

from __future__ import division
import numpy as np
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

__author__ = 'Alejandro Fontal'

curr_dir = os.getcwd()
seqdir = curr_dir + "/seqs/"
seqfiles = os.listdir(seqdir)
aa_string = "ARNDCEQGHILKMFPSTWYVX"  # 20 aa
props_file = "aa_propierties.csv"
add_props = False
seq_len = 1000

def get_1h_dict(aa_string, props_file, add_props=True):
    """
    Given a string of unique characters, generates dictionary of 1-hot vectors
    with characters as keys and vectors as values.

    """

    aa_dict = {}

    for idx, aa in enumerate(aa_string):

        if idx > 0:
            aa_dict[aa] = np.zeros(idx-1).tolist() + [1] +\
                          np.zeros(len(aa_string)-idx).tolist()
        else:
            aa_dict[aa] = [1] + np.zeros(len(aa_string)-1).tolist()

    if add_props:
        with open(props_file) as csvfile:
            aa_props = csv.reader(csvfile)
            for idx, row in enumerate(aa_props):
                if idx > 0:
                    aa = row[0]
                    props = row[1:]
                    aa_dict[aa] = map(float, props)

    return aa_dict




def seq2onehot(seq, aa_dict):
    """
    Takes a sequence(string) of length n and a dictionary with m keys and
    converts it to a 2 dimensional vector of nxm.

    """
    onehot = []

    # Convert ambiguous amino acids into actual ones.
    for aa in seq:

        if aa == "Z":
            aa = "Q"
        elif aa == "B":
            aa = "R"
        elif aa == "J":
            aa = "L"

        onehot += [aa_dict[aa]]

    return onehot


def fasta_process(fasta_fn, seq_len):
    """
    :param fasta_fn: Filename of the FASTA file to parse
    :return: A list containing the sequences in the FASTA file.
    """
    with open(fasta_fn) as fasta_file:
        fasta_list = fasta_file.read().splitlines()

        parsed_seqs = []
        for line in fasta_list:
            if line.startswith(">"):
                pass

            else:
                l = len(line)
                if l > 1000:
                    # Keep the N- and C- Terminal ends
                    parsed_seqs.append(line[0:seq_len//2] + line[l-seq_len:l])

                else:
                    # Add X's in the middle, keeping the ends.
                    half = l // 2
                    res = l % 2
                    parsed_seqs.append(line[0:half+res] + "X"*(seq_len-l) +
                                       line[l-half:])

    return parsed_seqs


class DataSet:
    def __init__(self, seqdir, props_file, seq_len=1000):
        self.seqfiles = os.listdir(seqdir)
        self.aa_string = "ARNDCEQGHILKMFPSTWYVX"  # 20 aa + X
        self.aa_dict = get_1h_dict(self.aa_string, props_file,
                                   add_props=add_props)
        self.labels = [i.replace(".fasta", "") for i in self.seqfiles]
        self.all_seqs = [fasta_process(seqdir + x, seq_len)
                         for x in self.seqfiles]
        self.train_seqs = []
        self.test_seqs = []
        self.train_tensor = []
        self.test_tensor = []

        for seqs in self.all_seqs:  # for seqs belonging to one specific label
            train_n = int(round(0.8 * len(seqs), 0))
            train_idxs = np.random.choice(len(seqs), train_n, replace=False)
            test_idxs = list(set(range(len(seqs))) - set(train_idxs))
            self.train_seqs.append([seqs[i] for i in train_idxs])
            self.test_seqs.append([seqs[i] for i in test_idxs])

        for idx, sub_loc in enumerate(self.train_seqs):
            sublabel = np.zeros(len(self.train_seqs))
            sublabel[idx] = 1

            for seq in sub_loc:
                self.train_tensor.append(
                    (sum(seq2onehot(seq, self.aa_dict), []), sublabel))

        for idx, sub_loc in enumerate(self.test_seqs):
            sublabel = np.zeros(len(self.test_seqs))
            sublabel[idx] = 1
            sub_list = []

            for seq in sub_loc:
                sub_list.append(
                    [sum(seq2onehot(seq, self.aa_dict), []), sublabel])

            self.test_tensor.append(sub_list)

        self.test_dict = {}
        for idx, label in enumerate(self.labels):
            self.test_dict[label] = self.test_tensor[idx]

        print len(self.test_tensor)

        self.test_tensor = [seq for subloc in self.test_tensor for seq in
                            subloc]

        print len(self.test_tensor)

    def print_labels(self):
        for label in self.labels:
            print label


# print "Sequences in training set: {}".format(len(train_tensor))
# print "Sequences in test set: {}\n".format(len(test_tensor))


