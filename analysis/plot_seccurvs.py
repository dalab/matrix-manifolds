import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import remove_extensions

parser = argparse.ArgumentParser(description='Plot graph sectional curvatures')
parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='The input seccurvs array file.')
args = parser.parse_args()

samples = np.load(args.input)
plt.hist(samples, bins=20, ec='k', density=True)
plt.xlabel('Sectional curvature')
plt.ylabel('Relative Frequency')
plt.title('dataset = {}'.format(remove_extensions(args.input)))

plt.tight_layout()
plt.savefig(args.input.replace('npy', 'png'))
