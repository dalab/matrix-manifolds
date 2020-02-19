import argparse

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot SPD sectional curvatures')
parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='The input seccurvs array file.')
args = parser.parse_args()

samples = np.load(args.input)
samples = samples[np.where(samples > np.percentile(samples, 5))]

plt.hist(samples, bins=20, ec='k', density=True)
plt.xlabel('Sectional curvature')
plt.ylabel('Relative Frequency')

plt.tight_layout()
plt.savefig(args.input.replace('npy', 'png'))
