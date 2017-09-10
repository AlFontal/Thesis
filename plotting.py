#!/usr/bin/env python
import matplotlib
matplotlib.use('GTKAgg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import matplotlib.patches as mpatches
from sys import argv


def plot_xents(name, col1="darkblue", col2="darkred", ylim=0.2):
    """

    Plot comparison for Neural Network perturbation
    """

    mean5 = np.loadtxt("mean5_{}.txt".format(name))
    mean15 = np.loadtxt("mean15_{}.txt".format(name))
    pos1 = np.arange(1, len(mean5)+1)
    pos2 = np.arange(1, len(mean15)+1)

    sns.set_style("darkgrid")
    plt.plot(pos1, mean5, col1, pos2, mean15, col2, linewidth=2.0)

    plt.xlabel("Position in sequence")
    plt.ylabel("Increase in CrossEntropy")
    plt.title("CrossEntropy perturbation per position in {} sequences".format(
        str(name)))

    plt.ylim([0, ylim])
    patch1 = mpatches.Patch(color=col1, label='window = 5 aa')
    patch2 = mpatches.Patch(color=col2, label='window = 15 aa')
    plt.legend(handles=[patch1, patch2], loc=1)

    plt.show()

if __name__ == "__main__":

    plot_xents(argv[1], ylim=0.25)
