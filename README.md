# Random ReLU Method
This repository maintains the code comparing the performance of random ReLU features to that of random Fourier features. It generates the experiment results for the paper [Random ReLU Features: Universality, Approximation, and Composition](https://arxiv.org/abs/1810.04374).

## Data Sets
3 real data sets and 4 synthetic data sets are tested in the experiment. The three data sets, mnist, adult and covtype, can be downloaded from the links listed in the paper. The synthetic data sets are generated using functions in `libsyndata.py`.

## Models and Experiments
The mathematical description of random ReLU features and random Fourier Features can be found in the paper. The code is in the file `librf.py`, written with tensorflow module. Wrappers of test steps are provided in `experiments.py`. Functions in `result_show.py` help print or plot results in a nice format. All the experiments are computed using University of Michigan Flux HPC Cluster.
