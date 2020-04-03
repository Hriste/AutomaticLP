#!/usr/bin/bash

module restore 
module load python/3.7-anaconda-2019.03
module load cuda/9.0

git clone https://github.com/tensorflow/models

conda deactivate

cd ../..

conda create -p ./.conda/envs/LPEnvironment 
conda init bash

conda activate LPEnvironment

conda install tensorflow-gpu=1.*
conda install ipykernel
conda install pillow
conda install matplotlib

ipython kernel install --user --name=LPKernel

