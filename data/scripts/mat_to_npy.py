import argparse
import os

import numpy as np
import scipy.io
from scipy.spatial.distance import squareform

parser = argparse.ArgumentParser(description='Transform the dissimilarity data '
                                 'from TU Delft from the .MAT format to numpy '
                                 'pairwise distances.')
parser.add_argument(
        '--input', type=str, required=True, help='The input .MAT file.')
parser.add_argument(
        '--output_dir', type=str, required=True, help='The output directory.')
args = parser.parse_args()

d = scipy.io.loadmat(args.input)['d']

dists = d['data'][0, 0]
assert len(np.where(np.abs(dists - dists.T) > 1e-6)[0]) == 0
pdists = np.clip(squareform(dists, checks=False), a_min=1e-4, a_max=None)
labels = d['nlab'][0, 0].flatten()

basename = os.path.splitext(os.path.basename(args.input))[0]
np.save(os.path.join(args.output_dir, basename) + '.npy', pdists)
np.save(os.path.join(args.output_dir, basename) + '-labels.npy', labels)
