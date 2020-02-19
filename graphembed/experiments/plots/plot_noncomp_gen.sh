#!/bin/bash

# degrees
python -m experiments.random.noncomp.plot_grid --root_dir runs/noncomp-r5-oct18 --plots degrees --num_cpus 2 --xmax 7.6 --ymax 1000
# seccurvs
python -m experiments.random.noncomp.plot_grid --root_dir runs/noncomp-r5-oct18 --plots seccurvs --num_cpus 2 --xmax 7.6 --ymin -0.85 --ymax 0.5
# angle ratios
python -m experiments.random.noncomp.plot_angle_ratios --root_dir runs/noncomp-r5-oct18 --dims 3 --manifolds hyp_4 spd_2 --ymax 20
python -m experiments.random.noncomp.plot_angle_ratios --root_dir runs/noncomp-r5-oct18 --dims 6 --manifolds hyp_7 spd_3 --ymax 20
python -m experiments.random.noncomp.plot_angle_ratios --root_dir runs/noncomp-r5-oct18 --dims 10 --manifolds hyp_11 spd_4 --ymax 20
