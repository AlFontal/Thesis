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


cyto_seqs = fasta_process("seqs/cytoplasmic.fasta")
nuclear_seqs = fasta_process("seqs/nuclear.fasta")
ER_seqs = fasta_process("seqs/ER.fasta")
extra_seqs = fasta_process("seqs/extracellular.fasta")
golgi_seqs = fasta_process("seqs/Golgi.fasta")
lyso_seqs = fasta_process("seqs/lysosomal.fasta")
mito_seqs = fasta_process("seqs/mitochondrial.fasta")
perox_seqs = fasta_process("seqs/peroxisomal.fasta")
plasma_seqs = fasta_process("seqs/plasma_membrane.fasta")
vacu_seqs = fasta_process("seqs/vacuolar.fasta")



all_seqs = [cyto_seqs] + [nuclear_seqs] + [ER_seqs] + [extra_seqs] +\
           [golgi_seqs] + [lyso_seqs] + [mito_seqs] + [perox_seqs] +\
           [plasma_seqs] + [vacu_seqs]



total_tensor = []

for idx, sub_loc in enumerate(all_seqs):

    sublabel = np.zeros(len(all_seqs))
    sublabel[idx] = 1

    for seq in sub_loc:
        total_tensor.append((sum(seq2onehot(seq), []), sublabel))


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