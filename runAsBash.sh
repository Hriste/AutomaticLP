#!/bin/bash
#SBATCH -N 1
#SBAtCH -n 6
#SBATCH -p gpuk80
#SBATCH --gres=gpu:1
#SBATCH -o tensorflow-%j.out
#SBATCH -t 0:30:0

module restore
module load cuda/9.0
module load anaconda

conda activate tensorflow

cd models/research
FILE=protobuf.zip
if [ ! -f "$FILE" ]; then
    wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
    unzip protobuf.zip
fi
./bin/protoc object_detection/protos/*.proto --python_out=.

pip install .
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../..

python ScriptForBashFile.py

mkdir -p trainingOutput
cd models/research
source activate tensorflow && python object_detection/legacy/train.py --logtostderr --train_dir=../../trainingOutput/ --pipeline_config_path=../../FromScratch/models/model/ssd_mobilenet_v1_coco.config
cd ../..

mkdir -p evalOutput
cd models/research
source activate tensorflow && python object_detection/legacy/eval.py --logtostderr --pipeline_config_path=../../FromScratch/models/model/ssd_mobilenet_v1_coco.config --checkpoint_dir=../../trainingOutput/ --eval_dir=../../evalOutput
cd ../..
