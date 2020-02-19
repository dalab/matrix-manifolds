import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from utils import annotate_vline, remove_extensions

parser = argparse.ArgumentParser(description='Plot delta-hyperbolicities')
parser.add_argument(
        '--input', type=str, required=True, help='The hyperbolicities file.')
args = parser.parse_args()

# Load the values.
values = np.load(args.input)
counts = np.load(args.input.replace('values', 'counts'))
counts = counts / np.sum(counts)
h_mean = np.sum(values * counts)
h_max = np.max(values)

# Plot the hyperbolicities.
plt.bar(values, counts, align='center', width=0.25, label='h (sampled)')
annotate_vline(h_mean, f'Mean: {h_mean:.2f}', color='tab:orange', lw=2)
annotate_vline(h_max, f'Max: {h_max:.2f}', left=False, color='r', lw=2)
plt.xlim([-0.5, h_max + 0.5])
plt.xlabel('Hyperbolicity')
plt.ylabel('Relative Frequency')
plt.title('dataset = {}'.format(remove_extensions(args.input)), y=1.08)

# Save the plot.
plt.tight_layout()
plt.savefig(args.input.replace('hyp-values.npy', 'hyperbolicity.png'))
