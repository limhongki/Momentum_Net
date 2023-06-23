# Momentum_Net

This repository is for reproducing following paper
about the *Momentum-Net* approach to image reconstruction
developed at the University of Michigan in 2019. 

If you use this repo,
please cite:

* Il Yong Chun, Zhengyu Huang, Hongki Lim, and Jeffrey Fessler:
  "Momentum-Net: Fast and convergent iterative neural network for inverse problems."
  [IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4915-31, Apr. 2023](https://doi.org/10.1109/TPAMI.2020.3012955).
* This paper was accepted in July 2020 and
  [here is the 2019 arXiv version of paper](https://arxiv.org/abs/1907.11818).


## Setting up and Reproducing

To reproduce the paper results,
please do the following:

* Install
  Michigan Image Reconstruction Toolbox (MIRT):
  http://web.eecs.umich.edu/~fessler/code/index.html

* Download CT reconstruction data used in the paper:
  https://www.dropbox.com/s/9kwl6zope87mje6/data.zip?dl=0

* Modify paths in `pcodes_init.m` and `train_iy.py` in `mypcodes` folder.

* Run `main_maptorch_ctrecon_ldEst_DeltaEst.m`.
