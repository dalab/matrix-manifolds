import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from graphembed.modules import BatchedObjective
from graphembed.train import TrainingEngine as Base

logger = logging.getLogger(__name__)


class TrainingEngine(Base):

    def __init__(self, min_alpha, max_alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.alpha is not None:
            raise ValueError('`alpha` param is not used in DA version of the '
                             'training engine.')
        if self.burnin_epochs or self.burnin_lower_lr or self.burnin_higher_lr:
            raise ValueError('Burnin is not supported in the DA version of the '
                             'training engine.')
        self.min_alpha = min_alpha
        self.update_alpha_every_epochs = self.n_epochs // \
                int(np.log2(max_alpha / min_alpha))

    def __call__(self, graph_dataset, last_step=0):
        # the parallelized objective function expecting node indices and other
        # objective-specific args (e.g, alpha)
        self.batched_obj = torch.nn.DataParallel(
                BatchedObjective(self.objective_fn, graph_dataset, self.emb))

        # training initialization
        self.pending_metric_results = []
        self.pending_metric_idx = 0
        self.global_step = last_step
        self.writer = SummaryWriter(log_dir=self.save_dir)

        alpha = self.min_alpha
        for epoch in range(1, self.n_epochs + 1):
            self._run_epoch(graph_dataset, alpha, epoch)

            if epoch % self.update_alpha_every_epochs == 0:
                alpha = 2 * alpha
                logger.info('New alpha: %.5f', alpha)

        # wait for all async computations
        self._consume_pending_metric_results(wait=True)
