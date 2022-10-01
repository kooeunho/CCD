# CCD
Centerized Clusters Distribution

We propose CCD as a novel non-linear filtering method for the input data. Including the Bitcoin price time-series, many financial indices suffer from the extreme bimodality. That is, the distribution with a high frequency in both ends of its PDF. Because the frequency is focused in both ends and the architecture must be trained to achieve a decent prediction performance on the ends, the architecture requires a high number of parameters, which can lead to overfitting as well as a low prediction performance on the ends. To overcome the challenge, we propose an input data filtering approach that mitigate the extreme bimodality by relocating high frequency region from both ends to the center (or mean) of the PDF. Also, we apply WES loss function to obtain further performance gain.

python version 3.6.9 tensorflow version 1.14.0

importing:

import numpy as np

import tensorflow as tf

%matplotlib inline

import os

from PyEMD import EMD, EEMD

from vmdpy import VMD

First, run "EEMD, VMD, SSA decomposition data generation.ipynb" to obtain "n_eemd.txt", "v_eemd.txt", "n_vmd.txt", "v_vmd.txt", "n_ssa.txt", and "v_ssa.txt".
