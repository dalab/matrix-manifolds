# Computationally Tractable Riemannian Manifolds for Graph Embeddings
This repository is the official implementation of [Computationally Tractable
Riemannian Manifolds for Graph Embeddings](https://arxiv.org/abs/2002.08665).

```
@article{cruceru2020computationally,
  title={Computationally Tractable Riemannian Manifolds for Graph Embeddings},
  author={Cruceru, Calin and B{\'e}cigneul, Gary and Ganea, Octavian-Eugen},
  journal={arXiv preprint arXiv:2002.08665},
  year={2020}
}
```


## Requirements
To reproduce the development environment, execute the following commands:

```setup
conda env create -f environment.yml
conda activate graphembed
env -i bash -l -c '/usr/bin/python setup.py develop --user'
```

Note that the last command requires a system-wide python distribution compiled
with GCC >= 9. It installs the Cython extension from
`graphembed/pyx/precision.pyx`. We include the pre-compiled `.so` file which
might work with other x86-64 Linux distribution, making it an optional step.

We used this workaround because at the time this code was written the latest
python distributions from conda was compiled with an older version of GCC which
does not support some of the C++17 features that we use in
`graphembed/pyx/impl/precision.cpp`.


## Training
To embed the graphs shown in Table 2, the following command is run:

```train
python -m experiments.run_grid \
  --datasets ../data/facebook.edges.gz ../data/web-edu.edges.gz ../data/bio-diseasome.edges.gz ../data/power.edges.gz \
  --loss_fns stress dist_1 dist_2 sne-incl_10 sne-incl_50 sne-excl_10 sne-excl_50 \
  --manifold euc_3 hyp_4 spd_2 spdstein_2 euc_6 hyp_7 spd_3 spdstein_3 \
  --eval_every 50 \
  --save_dir runs/nonpositive-curvature \
  --random_seed 42
```

This will train 3 separate embeddings for each configuration, for 3 different
optimization settings. The corresponding (hyper-) parameters can be seen in
`experiments/run_grid.py`.

Using CUDA devices will significantly speed up training. They will be
automatically detected and used if the environment variable
`CUDA_VISIBLE_DEVICES` is set appropriately.


## Evaluation
The reconstruction performance of each trained embedding is aggregated by
running the following command:

```eval
python -m experiments.agg_grid_results --root_dir runs/nonpositive-curvature --results_file nonpos-results.csv
```


## Results
The full results that correspond to those aggregated in Tables 5, 6, 7, are
included in the following Google spreadsheets:
* [non-positively curved spaces](https://docs.google.com/spreadsheets/d/1FsGuOiYCwoKUvWP1GeVu25Y4uE9sWGpePBAX8hwJTaM/edit#gid=0)
* compact spaces: [graphs](https://docs.google.com/spreadsheets/d/1APoC4r1F7LUmwZSpTki75PRkpt6abXXVej1xAglWqFY/edit?usp=sharing) and [dissimilarity matrices](https://docs.google.com/spreadsheets/d/1zOLBjPybr6pvaf2RPcbU0irwpXFOlXM0tfydQbX7Ro4/edit?usp=sharing)
* [Cartesian products](https://docs.google.com/spreadsheets/d/14SzV8r05FDcWoEzirgDKlVAKayeksTG7Udm8fBa6ELY/edit?usp=sharing)
