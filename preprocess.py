#!/usr/bin/env python

from __future__ import division
import numpy as np
import os
__author__ = 'Alejandro Fontal'

curr_dir = os.getcwd()
seqdir = curr_dir + "/seqs/"
seqfiles = os.listdir(seqdir)
aa_string = "NILVFMCAGOTSYWQBHEDKRP"  # 20 aa + N


def get_1h_dict(aa_string):
    """
    Given a string of unique characters, generates dictionary of 1-hot vectors
    with characters as keys and vectors as values.

    """

    aa_dict = {}

    for idx, aa in enumerate(aminoacids):

        if idx > 0:
            aa_dict[aa] = np.zeros(idx-1).tolist() + [1] +\
                          np.zeros(len(aa_string)-idx).tolist()
        else:
            aa_dict[aa] = [1] + np.zeros(len(aa_string)-1).tolist()

    return aa_dict


def seq2onehot(seq, aa_dict):
    """
    Takes a sequence(string) of length n and a dictionary with m keys and
    converts it to a 2 dimensional vector of nxm.

    """
    onehot = []
    for aa in seq:

        if aa == "Z":
            aa = "Q"
        elif aa == "B":
            aa = "R"
        elif aa == "J":
            aa = "L"
        elif aa == "X":
            aa = "N"

        onehot += [aa_dict[aa]]

    return onehot


def fasta_process(fasta_fn):
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
                parsed_seqs.append(line[0:100])

    return parsed_seqs


all_seqs = []

for file in seqfiles:
    all_seqs.append(fasta_process(seqdir+file))


total_tensor = []

aa_dict = get_1h_dict()

for idx, sub_loc in enumerate(all_seqs):

    sublabel = np.zeros(len(all_seqs))
    sublabel[idx] = 1

    for seq in sub_loc:
        total_tensor.append((sum(seq2onehot(seq, aa_dict), []), sublabel))


print len(total_tensor)

idxs = []

for i in range(len(total_tensor)):
    if len(total_tensor[i][0]) == 2200:
        idxs.append(i)

total_tensor = [total_tensor[i] for i in idxs]

train_idxs = np.random.choice(len(total_tensor), 0.8 * len(total_tensor),
                              replace=False)
test_idxs = list(set(range(len(total_tensor))) - set(train_idxs))

train_tensor = [total_tensor[i] for i in train_idxs]
test_tensor = [total_tensor[i] for i in test_idxs]

"""
class LocDataset()
"""
