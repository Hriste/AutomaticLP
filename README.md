# AutomaticLP
Automatic License Plate Generation and Detection 

![Diagram](Diagram.png)

## Top Level Directory Guide
LPGenerator.py - Python Script to generate image sets of randomized Maryland License Plate Sequences

MatlabImplementation - All Matlab Development Code

FromScratch - Python (Keras/Tensorflow) Development and Testing

object_detection - Copy of Tensorflow Object Detection API code

TestSet - folder containing test images for older Keras implmentation (To be depreciated)

## Running on MARCC
### Installation / Setup
1. Clone this repository `git clone https://github.com/Hriste/AutomaticLP.git`
2. In the top level of this repository clone the tensorflow object detection repository `git clone https://github.com/tensorflow/models`
3. Modify the ~./bashrc file with the following 2 lines where pwd is the path to the tensorflow object detection repository models/research folder. 
```
export PYTHONPATH=$PYTHONPATH:`pwd`
export PYTHONPATH=$PYTHONPATH:`pwd`/slim
```

4. From the top level of the AutomaticLP Repository load the following modules
```bash
module restore
module load python/3.7-anaconda-2019.03
module load cuda/9.0
```
5. Setup a conda enviroment with tensorflow (and tensorflow-gpu) installed. We need a version 1 install of tensorflow (not 2.*).
```bash
conda create --name LPEnviroment
conda install tensorflow-gpu=1.*
```
6. Add the conda enviroment to jupyter lab
```bash
conda active LPEnviroment
conda install ipykernel
ipython kernel install --user --name=LPKernel
```

### Train
7. Run jupyter lab on a partition with a gpu - I usually use the following 
```bash
sbatch -p debug -c 6 --gres=gpu:1 -t 2:0:0 jupyter_marcc lab
```

8. Launch the *GPUTraining* notebook
    - Make sure to select the LPKernel.
    - Compared to the *FullImplememtation* notebook this notebook pre-assummes some installation (previous steps), has some checks tests that the GPU is being used for training and simplifies the TF Record generation process through a python script (ScriptForBashFile - this needs to be renamed)

9. Running the notebook will populate the trainingOutput directory - this directory can then be used with tensorboard to evaluate results. 

(NOTE: as of 3/23/20 I can't run tensorboard while on MARCC for now I pushed the results to git under the firstTrial branch to analyze on my local PC, in contact with MARCC help desk to resolve) 

## To Do
- [ ] Add details about configuration file fields to modify
- [ ] Make a outline / list of configuration file fields to / can be modify as part of the tuning process
- [ ] Make a custom config file for this project
- [ ] Modify TF Record generator to accept 0 class (tensorflow dosen't like 0 being a class index)
- [ ] Clean up git repository
- [ ] Add Evaluation (either to existing notebook, new notebook, or as bash script)
- [ ] Update Diagram with new file names

## Notes & Refrences

### Tensorflow
We need a version 1 installation of tensorflow NOT 2.*. 

If you have tensorflow already installed check the version by running python in a terminal window and running: 

    import tensorflow as tf
    print(tf.__version__)
    
If your version of tensorflow is 2.* run 'pip uninstall tensorflow'.
If you don't have tensorflow installed or have just unistalled a 2.* version run 'pip install tensorflow=1.*'

(I am working with tensorflow 1.15.2)

On MARCC we're using a conda enviroment so instead of using pip uninstall / install use conda install 

 ### Configuration File
 Using the tensorflow object detection API training and evaluation is driven by a configuration file. 
 
 The configuration file is located in *models/model/ssd_mobilenet_v1_coco.config*

### Evaluation
(Images (TFRecord) for evaluation are specified in the configuration file)

Run the following from the object_detection directory:

    python legacy/eval.py \
    --logtostderr \
    --pipeline_config_path=../FromScratch_models/model/ssd_mobilenet_v1_coco.config \
    --checkpoint_dir=<path to training output directory> \
    --eval_dir=<path to evaluation output director>

 
My evaluation output directory is *object_detection/eval_output*
 
Use tensorboard to view the results of the evaluation 

    tensorboard --logdir=<path to evaluation output directory>
    
### Tensorboard Outputs

##### Scalars
Scalar graphs available include: 
- Learning Rate
- mAP (mean average percision). 
    - mAP is a value between 0-100, typically the higher is better
    - Each bounding box in an image is associated with a score. Based on this score precision-recall (PR curve) is computed by varying the score threshold. The average percision (AP) is the area under the PR curve. The AP is computed for each class and then averaged resulting in the mAP. 
    - A detection is a true positive if it has an 'intersection pver union' (IoU) with a ground truth box greater than some threshold (mAP@0.5 - the 0.5 refers to the threshold value)

#### Images

![Sample output](tensorboardWorking.png)
(note very little training was done for this but I wanted to put a sample here)

## Refrences
Training Procedure based on [THIS](https://github.com/tensorflow/models/blob/fae6ca34c3d7aab1aff0588bab6bd467e51ef13b/research/object_detection/g3doc/running_locally.md)

Legacy Training (train.py) based on [THIS](https://pythonprogramming.net/testing-custom-object-detector-tensorflow-object-detection-api-tutorial/?completed=/training-custom-objects-tensorflow-object-detection-api-tutorial/)
and [THIS](https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85)

[Exporting Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md)

[Installation Instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

[Tutorial 1] (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#monitor-training-job-progress-using-tensorboard)

[Tutorial 2](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73)
