import torch
from torch.utils.data import Dataset

from graphembed.utils import squareform1, triu_mask


class GraphDataset(Dataset):

    def __init__(self, pdists):
        # We store the max-normalized squared distances.
        pdists = pdists.pow(2)
        pdists.div_(pdists.max())
        self.pdists = squareform1(pdists)

    @property
    def device(self):
        return self.pdists.device

    def __getitem__(self, node_indices=None):
        if node_indices is None:
            pdists = self.pdists
        else:
            node_indices = node_indices.to(self.device)
            pdists = self.pdists[node_indices][:, node_indices]

        mask = triu_mask(len(pdists), d=1, device=self.device)
        return pdists.masked_select(mask)

    def __len__(self):
        return len(self.pdists)
