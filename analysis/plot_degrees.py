import argparse

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import remove_extensions

parser = argparse.ArgumentParser(description='Plot degrees')
parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='The input degrees array file.')
args = parser.parse_args()

degrees = np.load(args.input)
qs = [50, 75, 95, 99]
percentiles = np.percentile(degrees, qs)
mean = np.mean(degrees)
mean_color = 'cyan'
percentile_colors = ['y', 'm', 'r']

# Ignore outliers in the plot.
degrees = degrees[np.where(degrees <= percentiles[-1])]

bins = np.arange(1, max(degrees) + 1) - 0.5
plt.hist(degrees, bins=bins, density=True, ec='k', lw=0.5)
patches = [mpatches.Patch(color=mean_color, label='mean')]
patches += [
        mpatches.Patch(color=percentile_colors[i], label='{}%'.format(qs[i]))
        for i in range(len(qs) - 1)
]
plt.axvline(mean, color=mean_color, lw=2)
for i in range(len(qs) - 1):
    plt.axvline(percentiles[i], color=percentile_colors[i], lw=2)
plt.xlabel('Node degree')
plt.ylabel('Frequency')
plt.title('dataset = {}'.format(remove_extensions(args.input)), y=1.10)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(
        handles=patches,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc='lower left',
        mode='expand',
        borderaxespad=0,
        ncol=4)

plt.tight_layout()
plt.savefig(args.input.replace('npy', 'png'))
