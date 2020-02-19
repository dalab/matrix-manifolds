#!/bin/bash

python -m experiments.plots.graph_percentiles --datasets ../data/csphd.edges.gz ../data/power.edges.gz ../data/bio-diseasome.edges.gz ../data/grqc.edges.gz ../data/california.edges.gz ../data/road-minnesota.edges.gz ../data/web-edu.edges.gz ../data/facebook.edges.gz ../data/bio-wormnet.edges.gz --properties seccurvs

python -m experiments.plots.graph_percentiles --datasets ../data/csphd.edges.gz ../data/power.edges.gz ../data/bio-diseasome.edges.gz ../data/grqc.edges.gz ../data/california.edges.gz ../data/road-minnesota.edges.gz ../data/web-edu.edges.gz --properties degrees --big_datasets ../data/facebook.edges.gz ../data/bio-wormnet.edges.gz
