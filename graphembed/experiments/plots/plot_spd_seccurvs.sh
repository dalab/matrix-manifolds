#!/bin/bash

python -m experiments.agg_spd_seccurvs --dim 3 --root_dir runs/reals-all-spd-seccurvs
python -m experiments.agg_spd_seccurvs --dim 6 --root_dir runs/reals-all-spd-seccurvs
