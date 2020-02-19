import copy
import logging
import math
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from graphembed import metrics
from graphembed.metrics import area_under_curve, pearsonr
from graphembed.modules import BatchedObjective
from graphembed.utils import (PLT_MUTEX, latest_path_by_basename_numeric_order,
                              check_mkdir, Timer)

logger = logging.getLogger(__name__)

# All attributes of the `TrainingEngine` class below with their default values.
# The first three of them are not optional (see below).
default_attrs_ = dict(
        embedding=None,
        optimizer=None,
        objective_fn=None,
        alpha=None,
        n_epochs=2000,
        batch_size=None,
        drop_last_n=50,
        burnin_epochs=None,
        burnin_lower_lr=False,
        burnin_higher_lr=False,
        perturb_every_epochs=None,
        stabilize_every_epochs=None,
        lr_scheduler=None,
        min_lr=None,
        metrics=None,
        main_metric_idx=None,
        lazy_metrics=None,
        val_every_epochs=None,
        save_metrics_every_epochs=None,
        save_every_epochs=None,
        save_dir=None,
        snapshot_path=None)


class TrainingEngine:

    def __init__(self, **kwargs):
        self.__dict__.update(default_attrs_)
        self.__dict__.update(kwargs)

        # SANITY CHECKS
        if not self.embedding or not self.optimizer or not self.objective_fn:
            raise ValueError('`embedding`, `optimizer`, and `objective_fn` '
                             'must be specified to construct a TrainingEngine.')
        if self.burnin_lower_lr and self.burnin_higher_lr:
            raise ValueError('`burnin_lower_lr` and `burnin_higher_lr` are '
                             'mutually exclusive.')

        # DEFAULTS:
        # - one optimizer
        if not isinstance(self.optimizer, (tuple, list)):  # For legacy code.
            self.optimizer = [self.optimizer]
        # - one lr scheduler
        if self.lr_scheduler is not None and \
                not isinstance(self.lr_scheduler, (tuple, list)):
            self.lr_scheduler = [self.lr_scheduler]
        # - no burn-in epochs
        if self.burnin_epochs is None:
            self.burnin_epochs = 0
        # - do not perturb
        if self.perturb_every_epochs is None:
            self.perturb_every_epochs = self.n_epochs + 1
        # - do not stabilize
        if self.stabilize_every_epochs is None:
            self.stabilize_every_epochs = self.n_epochs + 1
        # - Pearson R and the average distortion as the default metrics
        if self.metrics is None:
            self.metrics = ['pearsonr', 'average_distortion']
        # - the first in the metrics list as the main metric
        if self.main_metric_idx is None:
            self.main_metric_idx = 0
        # - do not evaluate
        if self.val_every_epochs is None:
            self.val_every_epochs = self.n_epochs + 1
        # - save metrics at the last validation
        if self.save_metrics_every_epochs is None:
            self.save_metrics_every_epochs = self.n_epochs // \
                    self.val_every_epochs * self.val_every_epochs
        # - save at the very end
        if self.save_every_epochs is None:
            self.save_every_epochs = self.n_epochs
        # - temporary save dir
        if self.save_dir is None:
            self.save_dir = tempfile.gettempdir()
            check_mkdir(self.save_dir)
            logger.info('The save dir is (%s)', self.save_dir)

        # load the states if the snapshot path is given
        if self.snapshot_path:
            self._load()

    def __call__(self, graph_dataset, last_step=0):
        # the parallelized objective function expecting node indices and other
        # objective-specific args (e.g, alpha)
        self.batched_obj = torch.nn.DataParallel(
                BatchedObjective(self.objective_fn, graph_dataset,
                                 self.embedding))

        # burn-in epochs at higher temperature
        if self.burnin_epochs > 0 and self.alpha is not None:
            self._burnin_pre()
            self._burnin(graph_dataset)
            self._burnin_post()

        # training initialization
        self.pending_metric_results = []
        self.pending_metric_idx = 0
        self.best_struct = dict(
                epoch=0, loss=1e8, embedding=copy.deepcopy(self.embedding))
        self.global_step = last_step
        self.writer = SummaryWriter(log_dir=self.save_dir)

        try:
            for epoch in range(1, self.n_epochs + 1):
                self._run_epoch(graph_dataset, self.alpha, epoch)
                if self._check_early_break():
                    logger.warning('Early breaking (epoch=%d)', epoch)
                    break
        finally:
            self._save_best()  # save the best embedding

        # wait for all async computations
        self._consume_pending_metric_results(wait=True)

    def _burnin_pre(self):
        self.embedding.burnin(True)
        self.global_step = 0
        self.writer = SummaryWriter(
                log_dir=os.path.join(self.save_dir, 'burnin'))
        for optim in self.optimizer:
            for group in optim.param_groups:
                if self.burnin_lower_lr:
                    group['lr'] /= 10
                elif self.burnin_higher_lr:
                    group['lr'] *= 10

    def _burnin(self, graph_dataset):
        for i, alpha in enumerate([self.alpha / d for d in range(4, 0, -1)]):
            logger.info(f'Running burn-in epochs with alpha={alpha:.5f}')
            start_epoch = i * self.burnin_epochs + 1
            end_epoch = (i + 1) * self.burnin_epochs + 1
            for epoch in range(start_epoch, end_epoch):
                _ = self._train(graph_dataset, alpha, epoch)

                if epoch % self.stabilize_every_epochs:
                    with Timer('stabilizing'), torch.no_grad():
                        self.embedding.stabilize()

    def _burnin_post(self):
        for optim in self.optimizer:
            for group in optim.param_groups:
                if self.burnin_lower_lr:
                    group['lr'] *= 10
                elif self.burnin_higher_lr:
                    group['lr'] /= 10
        self.embedding.burnin(False)

    def _run_epoch(self, graph_dataset, alpha, epoch):
        with Timer('training'):
            loss = self._train(graph_dataset, alpha, epoch)
        # TODO(ccruceru): This should normally be done on validation.
        self._lr_scheduler_step(loss, epoch)

        # check best
        with Timer('checking if better model'):
            self._check_best(loss, epoch)

        # perturb
        if epoch % self.perturb_every_epochs == 0:
            with Timer('perturbing'), torch.no_grad():
                self.embedding.perturb(1 / epoch)  # Quite arbitrary norm.
        # stabilize
        if epoch % self.stabilize_every_epochs == 0:
            with Timer('stabilizing'), torch.no_grad():
                self.embedding.stabilize()
        # evaluate
        if epoch % self.val_every_epochs == 0:
            with Timer('validating'), torch.no_grad():
                val_obj = self._validate(graph_dataset, epoch)
            # we piggy-back this here: consume the lazy metrics
            self._consume_pending_metric_results()
        # save
        if epoch % self.save_every_epochs == 0:
            self._save(epoch)

    def _train(self, graph_dataset, alpha, epoch):
        n_points = len(graph_dataset)
        if self.batch_size is None:
            bs = n_points
        else:
            bs = min(n_points,
                     self.batch_size * max(1, len(self.batched_obj.device_ids)))

        perm = torch.randperm(n_points)  # Relies on default placement!
        total_loss = 0
        for i in range(0, n_points, bs):
            indices = perm[i:(i + bs)]
            if len(indices) < self.drop_last_n:
                break

            loss = self.batched_obj(indices, alpha=alpha, epoch=epoch).sum()
            for optim in self.optimizer:
                optim.zero_grad()
            loss.backward()
            for optim in self.optimizer:
                optim.step()

            self.global_step += 1
            self.writer.add_scalar(
                    str(self.objective_fn), loss / len(indices),
                    self.global_step)

            total_loss += loss.item()

        logger.debug('epoch %d, train loss %.5f', epoch, total_loss / n_points)
        return total_loss

    def _validate(self, graph_dataset, epoch):
        gpdists = graph_dataset[None].sqrt()  # Not in-place!
        mpdists = self.embedding.compute_dists(None).sqrt_()

        # plot the dists against each other
        indices = np.random.choice(
                np.arange(len(gpdists)),
                size=min(len(gpdists), 10000),
                replace=False)
        g_sub = gpdists.cpu().numpy()[indices]
        m_sub = mpdists.cpu().numpy()[indices]
        with PLT_MUTEX:
            plt.scatter(g_sub, m_sub, alpha=.3)
            self.writer.add_figure('dists', plt.gcf(), epoch)
            plt.close()

        # embedding stats
        self.embedding.add_stats(self.writer, epoch)

        # lazy metrics
        if self.lazy_metrics:
            mpdists_np = mpdists.cpu().numpy()  # lazy metrics expect np arrays!
            for name, f in self.lazy_metrics.items():
                self.pending_metric_results.append((epoch, name, f(mpdists_np)))

        # the other metrics
        gpdists = gpdists.to(mpdists.device)
        val_obj = None
        for i, metric in enumerate(self.metrics):
            val = getattr(metrics, metric)(mpdists, gpdists)
            self.writer.add_scalar(metric, val, epoch)
            if i == self.main_metric_idx:
                val_obj = val

        logger.info('epoch %d, val obj %.5f', epoch, val_obj)
        return val_obj

    def _consume_pending_metric_results(self, wait=False):
        while self.pending_metric_idx < len(self.pending_metric_results):
            epoch, name, future = \
                    self.pending_metric_results[self.pending_metric_idx]
            if not wait and not future.done():  # only block if ``wait=True``
                break
            value = future.result(None if wait else 0)
            if isinstance(value, float):
                self.writer.add_scalar(name, value, epoch)
            else:  # It has to be a pair of 1D vectors: (means, stds)!
                means, stds = value

                # Plot them.
                x = np.arange(1, len(means) + 1)
                with PLT_MUTEX:
                    l, = plt.plot(x, means)  # plt.plot(x, np.cumsum(means) / x)
                    plt.fill_between(
                            x,
                            means - stds,
                            means + stds,
                            facecolor=l.get_color(),
                            alpha=0.2)
                    plt.ylim(0, 1.05)  # All our metrics are between [0, 1]!
                    self.writer.add_figure(name, plt.gcf(), epoch)
                    plt.close()

                # Add the area under its graph too, for easier monitoring.
                self.writer.add_scalar('AUC_{}'.format(name),
                                       area_under_curve(means)[0], epoch)
                # Save it.
                if epoch % self.save_metrics_every_epochs == 0:
                    path = os.path.join(self.save_dir, f'mean_{name}_{epoch}')
                    np.save(path, means)
                    path = os.path.join(self.save_dir, f'std_{name}_{epoch}')
                    np.save(path, stds)
            self.pending_metric_idx += 1

    def _lr_scheduler_step(self, loss, epoch):
        # Makes sure we do not mess up (again) the fact that
        # ``ReduceLROnPlateau`` is not a type of torch ``_LRScheduler`` and,
        # hence, it ``step()`` function behaves differently.
        if self.lr_scheduler:
            for lrs in self.lr_scheduler:
                if isinstance(lrs, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lrs.step(loss)
                else:
                    lrs.step(epoch)

    def _check_early_break(self):
        if self.min_lr is None:
            return False
        for optim in self.optimizer:
            for group in optim.param_groups:
                if group['lr'] > self.min_lr + 1e-10:
                    return False
        return True

    def _check_best(self, loss, epoch):
        if loss < self.best_struct['loss']:
            self.best_struct = dict(
                    epoch=epoch,
                    loss=loss,
                    embedding=copy.deepcopy(self.embedding))

    def _save(self, epoch):
        path = os.path.join(self.save_dir, f'embedding_{epoch}.pth')
        torch.save(self.embedding.state_dict(), path)

    def _save_best(self):
        epoch = self.best_struct['epoch']
        loss = self.best_struct['loss']
        embedding = self.best_struct['embedding']

        # save the best embedding
        path = os.path.join(self.save_dir, 'best_embedding.pth')
        torch.save(embedding.state_dict(), path)

        # save the best loss
        path = os.path.join(self.save_dir, 'best_loss_{}'.format(epoch))
        with open(path, 'w') as f:
            f.write(f'{loss:.6f}')

    def _load(self):
        pattern = os.path.join(self.snapshot_path, 'embedding_*.pth')
        path = latest_path_by_basename_numeric_order(pattern)
        self.embedding.load_state_dict(torch.load(path))


class SineLRScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, T_max, eta_max=1, last_epoch=-1):
        if isinstance(T_max, (list, tuple)):
            self.T_max = T_max
        else:
            self.T_max = [T_max for _ in range(len(optimizer.param_groups))]
        if isinstance(eta_max, (list, tuple)):
            self.eta_max = eta_max
        else:
            self.eta_max = [eta_max for _ in range(len(optimizer.param_groups))]
        super(SineLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
                base_lr + (self.eta_max[i] - base_lr) * \
                    (1 - math.cos(math.pi * self.last_epoch / self.T_max[i])) / 2
                for i, base_lr in enumerate(self.base_lrs)
        ]
