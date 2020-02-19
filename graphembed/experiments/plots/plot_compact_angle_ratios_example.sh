#!/bin/bash

python -m experiments.plot_angle_ratios --root_dir runs/spherical-sep --dims 4 --datasets drill_shaft_zip --loss_fns sne-excl_10 sne-excl_50 --manifolds sph_5 grass_4_2 --xmin -0.05 --xmax 1.05 --no_ylabel
python -m experiments.plot_angle_ratios --root_dir runs/spherical-sep --dims 4 --datasets drill_shaft_zip --loss_fns dist_2 --manifolds sph_5 grass_4_2 --xmin -0.05 --xmax 1.05

# python -m experiments.plot_angle_ratios --root_dir runs/spherical-sep --dims 4 --datasets bun_zipper_res3 --loss_fns dist_1 --manifolds sph_5 grass_4_2 --xmin -0.05 --xmax 1.05
# python -m experiments.plot_angle_ratios --root_dir runs/spherical-sep --dims 4 --datasets drill_shaft_zip --loss_fns dist_1 --manifolds sph_5 grass_4_2 --xmin -0.05 --xmax 1.05 --no_ylabel
# python -m experiments.plot_angle_ratios --root_dir runs/spherical-sep --dims 4 --datasets road-minnesota  --loss_fns dist_1 --manifolds sph_5 grass_4_2 --xmin -0.05 --xmax 1.05 --no_ylabel

# python -m experiments.plot_angle_ratios --root_dir runs/spherical-diss-oct21 --dims 4 --datasets catcortex --loss_fns sne-excl_10 sne-excl_50 --manifolds sph_5 grass_4_2 --xmin -0.05 --xmax 1.05 --no_ylabel
# python -m experiments.plot_angle_ratios --root_dir runs/spherical-diss-oct21 --dims 4 --datasets catcortex --loss_fns dist_2 --manifolds sph_5 grass_4_2 --xmin -0.05 --xmax 1.05
