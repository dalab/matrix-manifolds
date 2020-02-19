#!/bin/bash

# reals
python -m experiments.plot_run_grid --root_dir runs/reals-all --datasets csphd --loss_fns sne-incl_10  --dims 3 --no_right_ylabel --right_ymax 0.36 --leftmost --use_std_colors
python -m experiments.plot_run_grid --root_dir runs/reals-all --datasets facebook --loss_fns sne-incl_10  --dims 3 --no_right_ylabel --no_left_ylabel --right_ymax 0.36 --use_std_colors
python -m experiments.plot_run_grid --root_dir runs/reals-all --datasets power --loss_fns sne-incl_10  --dims 3 --no_left_ylabel --right_ymax 0.36 --use_std_colors
# noisy
python -m experiments.plot_run_grid --root_dir runs/reals-noisy --datasets csphd --flip_probabilities 0.0500  --loss_fns sne-incl_10  --dims 3 --no_right_ylabel --right_ymax 0.36 --use_std_colors
python -m experiments.plot_run_grid --root_dir runs/reals-noisy --datasets facebook --flip_probabilities 0.0100  --loss_fns sne-incl_10  --dims 3 --no_left_ylabel --no_right_ylabel --right_ymax 0.36 --use_std_colors
python -m experiments.plot_run_grid --root_dir runs/reals-noisy --datasets power --flip_probabilities 0.0100  --loss_fns sne-incl_10  --dims 3 --no_left_ylabel --right_ymax 0.36 --use_std_colors
