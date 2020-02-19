import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import remove_extensions

parser = argparse.ArgumentParser(description='Plot nodes per layer')
parser.add_argument('--input', type=str, required=True, help='The input file.')
args = parser.parse_args()

npl = np.load(args.input)
npl = npl / np.sum(npl)

plt.bar(np.arange(len(npl)) + 1, npl, ec='k', lw=0.5)
plt.xlabel('Layer')
plt.ylabel('Relative Frequency')
plt.title('dataset = {}'.format(remove_extensions(args.input)))
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(args.input.replace('npy', 'png'))
