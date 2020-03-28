#!/usr/bin/bash

module restore 
module load python/3.7-anaconda-2019.03
module load cuda/9.0

git clone https://github.com/tensorflow/models

conda deactivate

conda create --name LPEnviroment
conda init bash

conda activate LPEnviroment

conda install tensorflow-gpu=1.*
conda install ipykernel
conda install pillow
conda install matplotlib

ipython kernel install --user --name=LPKernel
