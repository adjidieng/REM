# REM

This repo provides code that accompanies the paper "Reweighted Expectation Maximization" by Dieng and Paisley, 2019 (). 
It uses pytorch 1.1.0

## Example command to run REM (v1) on omniglot:

python main.py --version v1 --dataset omniglot --epochs 200 --z_dim 20 --n_samples_train 1000 --n_samples_test 1000

The code for preprocessing the data is borrowed from https://github.com/yoonholee/pytorch-vae/tree/master/data_loader 
