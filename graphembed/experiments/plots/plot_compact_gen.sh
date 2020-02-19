#!/bin/bash

# degrees
python -m experiments.random.compact.plot_grid --root_dir runs/compact-gen-oct16/ --plots degrees --num_cpus 2 --xmax 4.5
# seccurvs
python -m experiments.random.compact.plot_grid --root_dir runs/compact-gen-oct16/ --plots seccurvs --num_cpus 2 --xmax 4.5 --ymin -0.3 --ymax 0.5
# angle ratios
python -m experiments.random.compact.plot_angle_ratios --root_dir runs/compact-gen-oct16 --dims 2 --manifolds sph_3 grass_3_1
python -m experiments.random.compact.plot_angle_ratios --root_dir runs/compact-gen-oct16 --dims 3 --manifolds sph_4 grass_4_1 so_3
python -m experiments.random.compact.plot_angle_ratios --root_dir runs/compact-gen-oct16 --dims 4 --manifolds sph_5 grass_5_1 grass_4_2
