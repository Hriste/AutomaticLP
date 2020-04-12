#!/usr/bin/bash


echo "This setup script assummes you've cloned the AutomaticLP Repo and are executing from user@jhu.edu/code/AutomaticLP."
echo "Affirm this is the case:"
select yn in "Yes" "No"; do
  case $yn in
    Yes ) break;;
    No ) return;;
  esac
done

module restore
module load python/3.7-anaconda-2019.03
module load cuda/9.0

git clone https://github.com/tensorflow/models

git clone https://github.com/Hriste/tf_object_detection_cm.git

echo "Overwrite current tf_example_parser with modifications."
cp tf_object_detection_cm/tf_example_parser.py models/research/object_detection/metrics/tf_example_parser.py

conda deactivate

cd ../..

conda create -p ./.conda/envs/LPEnvironment
conda init bash

conda activate LPEnvironment

conda install tensorflow-gpu=1.*
conda install ipykernel
conda install pillow
conda install matplotlib
conda install -c menpo opencv
conda install pandas
conda install scikit-learn

ipython kernel install --user --name=LPKernel
