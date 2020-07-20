import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import csv
import imageio
import random
import tensorflow as tf
import augmentor
import TFRecordConverter

'''
Christina Paolicelli
July 18th 2020

This script takes in the current "ideal" test directory and overwrites the
images with non-ideal versions via applying "augmentation"
'''

HEIGHT = 256
WIDTH = 512

thing, choosen = augmentor.select_augmentation()

# Handle the ALL case - not supporting here for now
if "all" in choosen.lower():
    print("All is not a supported option in this context")
    exit()

path = input("Enter the test image directory: ")
with open(path + "/dataset.csv") as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        new_row = row
        image_name = row[0]
        image = imageio.imread(path + "/" + image_name)

        if "affine" in choosen.lower():
            # some augmentors also need the bounding boxes updated
            image_aug, modified_row = augmentor.updateBoundingBoxes(new_row, augmentor, image)
            rowsToAdd.append(modified_row)
        else:
            ia.seed(1)
            image_aug = thing(image = image)

        # This is the overwite
        im_name = path + "/" + image_name
        imageio.imwrite(im_name, image_aug[:, :, 0])

    # If Affine (or bounding boxes modified need to overwrite the csv)
    if "affine" in choosen.lower():
        with open(path + "/dataset.csv", 'w') as fd:
            for row in rowsToAdd:
                stringToWrite = ""
                for entry in row:
                    stringToWrite = stringToWrite + entry + ","
                fd.write(stringToWrite)
                fd.write("\n")

# Now we have an updated Image Directory and CSV - need to update the TF Record
writer = tf.io.TFRecordWriter("FromScratch/TFRecordEval.tfrecord")

with open(path+"/dataset.csv") as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        tf_example = TFRecordConverter.createTFRecord(row, path)
        writer.write(tf_example.SerializeToString())

writer.close()
