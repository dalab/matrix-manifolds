#!/bin/bash

python -m experiments.plots.plot_noncomp_ar_boxplots_up   --root_dir runs/reals-all
python -m experiments.plots.plot_noncomp_ar_boxplots_down --root_dir runs/reals-all
