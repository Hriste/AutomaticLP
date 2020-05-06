#!/usr/bin/bash

module restore
module load python/3.7-anaconda-2019.03
module load cuda/9.0

# Get Tensorflow Object Detection API Code
git clone https://github.com/tensorflow/models

# Get Forked Confusion Matrix Project
git clone https://github.com/Hriste/tf_object_detection_cm.git

# Confusion Matrix Project requires a path for tensorflow
echo "Overwrite current tf_example_parser with modifications."
cp tf_object_detection_cm/tf_example_parser.py models/research/object_detection/metrics/tf_example_parser.py

# Install protobuf
cd models/research
FILE=protobuf.zip
if [ ! -f "$FILE" ]; then
    wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
    unzip protobuf.zip
fi
./bin/protoc object_detection/protos/*.proto --python_out=.
pip install --user .
cd ../..

# pip install for augmentation
pip install --user imagecorruptions

# Download and Extract models
cd FromScratch
source getModels.sh
cd ..
