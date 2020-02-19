python -m experiments.plots.plot_synthetics --root_dir runs/synthetic-oct22/ --datasets grid_n10_d3  --loss_fns sne-incl_10  --dims 3 --leftmost
python -m experiments.plots.plot_synthetics --root_dir runs/synthetic-oct22/ --datasets cycle1000  --loss_fns  dist_2  --dims 3
python -m experiments.plots.plot_synthetics --root_dir runs/synthetic-oct22/ --datasets btree1365  --loss_fns sne-incl_10  --dims 3
python -m experiments.plots.plot_synthetics --root_dir runs/synthetic-oct22/ --datasets treecycle1210_n10_h4_r3  --loss_fns sne-incl_10  --dims 3 --rightmost
