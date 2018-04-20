#!/usr/bin/env python
"""
Plot the distribution of the weights of a layer of the NN
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description="A simple utility to plot the distrubtion of weights of tensorpack networks.")
    parser.add_argument('-i','--input-file',type=str,
            help='The name of the input file (the dictionary of weights)', required=True)
    parser.add_argument('-o','--output-file',type=str,
            help='The name of the output file (the plot)', required=True)
    parser.add_argument('-k','--key',type=str, required=True,
            help='The name of the key from which to access the weights')
    parser.add_argument('-b','--bins',type=int, default=50,
            help='The number of bins to use in the histogram (default: %(default)s)')
    args = parser.parse_args()

    d = np.load(args.input_file).item()
    x = d[args.key]
    n, bins, patches = plt.hist(x.flatten(), args.bins, normed=1, alpha=0.75)
    plt.savefig(args.output_file, format='pdf')
