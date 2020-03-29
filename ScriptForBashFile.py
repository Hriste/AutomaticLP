# Christina Paolicelli
# 3/14/20
#
# IT IS ASSUMMED THIS IS BEING RUN FROM THE TOP LEVEL AutomaticLP directory
#
import LPImageGenerator
import TFRecordConverter
from datetime import datetime
from numpy import floor
import csv
import tensorflow as tf

CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
PATH_TO_LABELS = './data/label_map.pbtxt'

'''
GENERATE DATASETS
'''
## we need a training set and a test dataset
totalNumOfSamples = input("Enter the total number of samples (60% is training, 40% is testing): ")
trainDirectoryName = datetime.now().strftime("TrainingImages_%Y-%m-%d_%H-%M")
testDirectoryName = datetime.now().strftime("TestImages_%Y-%m-%d_%H-%M")
numTrainImages = int(floor(int(totalNumOfSamples)*.6))
numTestImages = int(totalNumOfSamples) - numTrainImages

LPImageGenerator.makeSequences(numTrainImages, trainDirectoryName)
print(numTrainImages, " Training Images Generated")

LPImageGenerator.makeSequences(numTestImages, testDirectoryName)
print(numTestImages, " Test Images Generated")


'''
CONVERT DATASETS to TF RECORDS
'''

# Generate TF Record for Training
writer = tf.io.TFRecordWriter("FromScratch/TFRecordTrain.tfrecord")
with open(trainDirectoryName+"/dataset.csv") as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        tf_example = TFRecordConverter.createTFRecord(row, trainDirectoryName)
        writer.write(tf_example.SerializeToString())

writer.close()

# Generate TF Record for Evaluation
writer = tf.io.TFRecordWriter("FromScratch/TFRecordEval.tfrecord")

with open(testDirectoryName+"/dataset.csv") as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        tf_example = TFRecordConverter.createTFRecord(row, testDirectoryName)
        writer.write(tf_example.SerializeToString())

writer.close()
