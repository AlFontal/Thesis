#!/usr/bin/env python

from __future__ import division
import numpy as np
import random

__author__ = 'Alejandro Fontal'

aminoacids = "NILVFMCAGOTSYWQBHEDKRP"

aa_dict = {}
for idx, aa in enumerate(aminoacids):

    if idx > 0:
        aa_dict[aa] = np.zeros(idx-1).tolist() + [1] +\
                      np.zeros(len(aminoacids)-idx).tolist()
    else:
        aa_dict[aa] = [1] + np.zeros(len(aminoacids)-1).tolist()


def seq2onehot(seq):

    onehot = []
    for aa in seq:

        if aa == "Z":
            aa = "Q"

        if aa == "B":
            aa = "R"

        if aa == "J":
            aa = "L"

        if aa == "X":
            aa = "N"

        onehot += [aa_dict[aa]]

    return onehot

def fasta_parse(fasta_fn):
    """
    :param fasta_fn: Filename of the FASTA file to parse
    :return: A dictionary containing labels as keys and sequences as values.
    """
    with open(fasta_fn) as fasta_file:
        fasta_list = fasta_file.read().splitlines()

        parsed_seqs = {}
        for line in fasta_list:
            if line.startswith(">"):
                label = line[1:]
                parsed_seqs[label] = ""

            else:
                parsed_seqs[label] += line[0:100]

    return parsed_seqs




cyto_seqs = fasta_parse("seqs/cytoplasmic.fasta")
nuclear_seqs = fasta_parse("seqs/nuclear.fasta")

cyto_tensor = []

for seq in cyto_seqs.values():
    cyto_tensor.append((sum(seq2onehot(seq), []), [1, 0]))

nuclear_tensor = []

for seq in nuclear_seqs.values():
    nuclear_tensor.append((sum(seq2onehot(seq), []), [0, 1]))

total_tensor = nuclear_tensor + cyto_tensor

idxs = []

for i in range(len(total_tensor)):
    if len(total_tensor[i][0]) == 2200:
        idxs.append(i)
total_tensor = [total_tensor[i] for i in idxs]

