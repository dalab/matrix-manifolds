#!/bin/bash

python -m experiments.plots.plot_hyp_spd_gap --root_dir runs/reals-all --dims 3 --datasets  facebook  --manifolds spdstein_2 spd_2 hyp_4 --show_legend
python -m experiments.plots.plot_hyp_spd_gap --root_dir runs/reals-all --dims 3 --datasets  power     --manifolds spdstein_2 spd_2 hyp_4 --no_ylabel
python -m experiments.plots.plot_hyp_spd_gap --root_dir runs/reals-all --dims 3 --datasets  web-edu   --manifolds spdstein_2 spd_2 hyp_4 --no_ylabel
