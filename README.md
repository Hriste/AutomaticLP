# AutomaticLP
Automatic License Plate Generation and Detection 
NOTE: the evaluation side of the pipeline is still under development

![Diagram](Diagram.png)

## Top Level Directory Guide
LPGenerator.py - Python Script to generate image sets of randomized Maryland License Plate Sequences

MatlabImplementation - All Matlab Development Code

FromScratch - Python (Keras/Tensorflow) Development and Testing

Archived - Old files to keep around, not under active development or use other than as refrence

## Running on MARCC
### Installation / Setup
1. Clone this repository `git clone https://github.com/Hriste/AutomaticLP.git` (if on MARCC use the /code folder)
2. In the top level of this repository clone the tensorflow object detection repository `git clone https://github.com/tensorflow/models`
3. Modify the ~./bashrc file with the following 2 lines where pwd is the path to the tensorflow object detection repository models/research folder. 
```
export PYTHONPATH=$PYTHONPATH:`pwd`
export PYTHONPATH=$PYTHONPATH:`pwd`/slim
```

4. From the top level of the AutomaticLP Repository run the setUpScript.sh  
If you have issues with this run the commands individually.  
*NOTE: the script assummes you're executing from the user@jhu.edu/code/AutomaticLP if this is not the case adjust the change directory commands in the script accordingly.*


### Train
Video Training Walkthrough [HERE](https://youtu.be/irMbEhf_J2o)

7. Run jupyter lab on a partition with a gpu - I usually use the following 
```bash
sbatch -p debug -c 6 --gres=gpu:1 -t 2:0:0 jupyter_marcc lab
```

8. Launch the *GPUTraining* notebook
    - Make sure to select the LPKernel.

9. Running the notebook will populate the trainingOutput directory - this directory can then be used with tensorboard to evaluate results.  
If you want a brand new training session delete any prexisting trainingOutput folder.  
*Each time you run reset the Kernel - otherwise there can be issues logging to file*

## To Do
[ ] Add Evaluation to pipeline
[ ] Add setup needed for evaluation to run (import thrid party projects, add packages, etc.)

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
