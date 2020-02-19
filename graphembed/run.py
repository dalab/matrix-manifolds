import logging
import logging.config
logging.config.fileConfig('logging.conf')

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from shutil import copyfile
import sys

import matplotlib
matplotlib.use('Agg')
import torch

from graphembed.data import load_graph_pdists, GraphDataset
from graphembed.products.embedding import Embedding as ProductManifoldEmbedding
from graphembed.pyx import FastPrecision
from graphembed.utils import check_mkdir, nnm1d2_to_n, Timer


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = parse_config(args.config)
    set_seeds(args.random_seed)
    save_dir = check_mkdir(config['save_dir_root'], increment=True)
    copyfile(args.config, os.path.join(save_dir, 'config.yaml'))

    # torch settings
    torch.set_default_dtype(torch.float64)  # use double precision
    if torch.cuda.is_available():  # place everything on CUDA
        # NOTE: We rely on this in several parts of the code.
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # prepare data
    gpdists, g = load_graph_pdists(
            config['input_graph'], cache_dir=config.get('cache_dir'))
    n_nodes = nnm1d2_to_n(len(gpdists))
    if 'preprocess' in config:
        gpdists = config['preprocess'](gpdists)
    dataset = GraphDataset(gpdists if n_nodes < 5000 else gpdists.to('cpu'))

    # the embedding
    embedding = config['embedding'](n_nodes)

    # the optimizers
    optimizers = []
    lr_schedulers = []
    if 'embedding_optimizer' in config:
        emb_optim = config['embedding_optimizer'](embedding.xs)
        optimizers.append(emb_optim)
        if 'embedding_lr_scheduler' in config:
            lr_schedulers.append(config['embedding_lr_scheduler'](emb_optim))
    if 'curvature_optimizer' in config:
        curv_optim = config['curvature_optimizer'](embedding.curvature_params)
        optimizers.append(curv_optim)
        if 'curvature_lr_scheduler' in config:
            lr_schedulers.append(config['curvature_lr_scheduler'](curv_optim))

    # prepare training
    training_args = dict(
            embedding=embedding,
            optimizer=optimizers,
            lr_scheduler=lr_schedulers,
            objective_fn=config['objective_fn'],
            save_dir=save_dir)
    training_args.update(config['training_params'])

    # use the right training engine
    if isinstance(embedding, ProductManifoldEmbedding):
        from graphembed.products import TrainingEngine
    elif 'min_alpha' in training_args or 'max_alpha' in training_args:
        from graphembed.train_da import TrainingEngine
    else:
        from graphembed.train import TrainingEngine

    # use a with-block to make sure we the threads are closed even if we kill
    # the process
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        if g is not None:
            with Timer('constructing FastPrecision', loglevel=logging.INFO):
                fp = FastPrecision(g)
            training_args['lazy_metrics'] = {
                'Layer_Mean_F1': \
                    lambda p: pool.submit(fp.layer_mean_f1_scores, p),
            }  # yapf: disable
        training_engine = TrainingEngine(**training_args)

        # train
        with Timer('training', loglevel=logging.INFO):
            training_engine(dataset)


def parse_args():
    parser = argparse.ArgumentParser(description='Graph embedding driver.')
    parser.add_argument(
            '--config',
            type=str,
            help='The YAML config which sets up this driver.')
    parser.add_argument(
            '--random_seed',
            type=int,
            default=42,
            help='The manual random seed.')
    parser.add_argument(
            '--num_workers',
            type=int,
            default=4,
            help='The number of workers used to compute lazy '
            '(i.e., slow) metrics.')
    parser.add_argument(
            '--detect_anomaly',
            action='store_true',
            help='Enable PyTorch anomaly detection')
    parser.add_argument(
            '--verbose',
            action='store_true',
            help='Sets the log level to DEBUG.')

    return parser.parse_args()


def set_seeds(seed):
    import random
    import numpy

    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def parse_config(config_path):
    r"""Parses a YAML config file into a Python dict with support for closures
    and object construction.
    """
    import copy
    from functools import reduce
    import importlib
    import operator
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    yaml = YAML()
    config = yaml.load(open(config_path, 'r'))
    classes = {}

    def get_from_dict(data_dict, map_list):
        return reduce(operator.getitem, map_list, data_dict)

    def set_in_dict(data_dict, map_list, value):
        get_from_dict(data_dict, map_list[:-1])[map_list[-1]] = value

    # assume if a function definition is in the most outer scope, it will not
    # have anywhere in config a nested class
    def walk(node, curr=None, depth=0):
        if curr is None:
            curr = {}
        if isinstance(node, CommentedMap):
            for key, item in node.items():
                curr[depth] = key
                if key in ['object', 'closure']:
                    if depth in classes.keys():
                        classes[depth].append(copy.deepcopy(curr))
                    else:
                        classes[depth] = [copy.deepcopy(curr)]
                walk(item, copy.deepcopy(curr), depth + 1)
        elif isinstance(node, CommentedSeq):
            for i, _ in enumerate(node):
                curr[depth] = i
                walk(node[i], copy.deepcopy(curr), depth + 1)

    walk(config)
    depths = classes.keys()
    depths = reversed(sorted(depths))
    for depth in depths:
        for keys_dict in classes[depth]:
            keys = list(keys_dict.values())
            item = get_from_dict(config, keys)
            parts = item['name'].split('.')
            module = importlib.import_module('.'.join(parts[:-1]))

            # get the constructor for this instance
            ctor = getattr(module, parts[-1])
            # unify branches by setting an empty dict if no params provided
            if 'params' not in item or item['params'] is None:
                item['params'] = {}
            if isinstance(item['params'], CommentedSeq):
                item['params'] = list(item['params'])

            try:
                if isinstance(item['params'], dict):
                    instance = ctor(**item['params'])
                elif isinstance(item['params'], list):
                    instance = ctor(*item['params'])
            except TypeError:
                # pylint: disable=cell-var-from-loop
                instance = lambda *args, f=ctor, params=item['params'], \
                        **kwargs: f(*args, **kwargs, **params)
                # pylint: enable=cell-var-from-loop
            set_in_dict(config, keys[:-1], instance)

    return config


if __name__ == '__main__':
    sys.exit(main())
