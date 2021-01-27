# Momentum_Net

This repository is for reproducing following paper
about the *Momentum-Net* approach to image reconstruction: 

Il Yong Chun, Zhengyu Huang, Hongki Lim, and Jeffrey Fessler:
"Momentum-Net: Fast and convergent iterative neural network for inverse problems."
[IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020. To appear](https://doi.org/10.1109/TPAMI.2020.3012955)

[arXiv version of paper.](https://arxiv.org/pdf/1907.11818)


## Setting up and Reproducing

To reproduce the paper, please make sure you have the following:
Michigan Image Reconstruction Toolbox (MIRT) installed:
http://web.eecs.umich.edu/~fessler/code/index.html.  

Modify paths in `pcodes_init.m` and `train_iy.py` in `mypcodes` folder.

Then run `main_maptorch_ctrecon_ldEst_DeltaEst.m`.
