#!/usr/bin/env python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Alejandro Fontal'

aminoacids = "NILVFMCAGOTSYWQBHEDKRP"

aa_dict = {}
for idx, aa in enumerate(aminoacids):
    if idx > 0:
        aa_dict[aa] = np.zeros(idx-1).tolist() + [1] +  np.zeros(22-idx).tolist()
    else:
        aa_dict[aa] = [1] + np.zeros(21).tolist()

def seq2onehot(seq):

    onehot = []
    for aa in seq:
        onehot += [aa_dict[aa]]

    return onehot


if __name__ == "__main__":


    seq1 = "MTSSDTQNNKTLAAMKNFAEQYAKRTDTYFCSDLSVTAVVIEGLARHKEELGSPLCPCRH"
    seq2 = "YEDKEAEVKNTFWNCPCVPMRERKECHCMLFLTPDNDFAGDAQDIPMETLEEVKASMACP"


onehot = seq2onehot(seq1)

for aa in onehot:
    print aa

print type(onehot)

