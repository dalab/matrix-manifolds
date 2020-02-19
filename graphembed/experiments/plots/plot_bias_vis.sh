#!/bin/bash

python -m experiments.random.old.gen_man_graph --manifold hyp_4 --radius 5 --num_nodes 1000
mv output/tmp/hyp_4/sample.pdf hyp-exp.pdf
python -m experiments.random.old.gen_man_graph --manifold hyp_4 --use_rs --radius 5 --num_nodes 1000
mv output/tmp/hyp_4/sample.pdf hyp-unif.pdf

python -m experiments.random.old.gen_man_graph --manifold sph_3 --radius 3.14 --num_nodes 1000
mv output/tmp/sph_3/sample.pdf sph-exp.pdf
python -m experiments.random.old.gen_man_graph --manifold sph_3 --use_rs --radius 3.14 --num_nodes 1000
mv output/tmp/sph_3/sample.pdf sph-unif.pdf

rm -r output/tmp
