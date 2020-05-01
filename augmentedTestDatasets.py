'''
Generate Augmented Test Datasets
Christina Paolicelli
5/1/20
'''
import sys, getopt
import os
import LPImageGenerator
import TFRecordConverter
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import csv


AUGMENTATION_TYPES = ["GaussianNoise"]

def apply_gaussian_noise(directory, filename):
    # https://theailearner.com/2019/05/07/add-different-noise-to-an-image/
    path = os.path.join(directory, filename)
    img = cv2.imread(path)

    # Generate Gaussian noise
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    # Add the noise to the image
    img_gauss = img + gauss#cv2.add(img, gauss)

    im = Image.fromarray(img_gauss)
    im.save(path)



def show_help():
    print("-h,                              Displays this help message")
    print("-n <num>                         Number of Images to generate")
    print("-a, augmentationType>            Applies the given augmentation type.")
    print("Valid types of augmentation are: ")
    for entry in AUGMENTATION_TYPES:
        print(entry)

def main(argv):
    print(argv)
    numImages = 0
    augmentationType = ""
    try:
        opts, args = getopt.getopt(argv[1:], "ha:n:")
    except getopt.GetoptError:
        sys.exit(2)

    print(opts)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            show_help
            sys.exit()
        if opt in ("-n", "--number"):
            numImages = arg
        if opt in ("-a", "--augmentation"):
            if arg in AUGMENTATION_TYPES:
                augmentationType = arg
            else:
                sys.exit()

    if augmentationType == "":
        print("No Augmentation specified")
        sys.exit()
    # Create Directory for augmented images
    augmentedDir = datetime.now().strftime(augmentationType + "Images_%Y-%m-%d_%H-%M")

    # Generate the images @ baseline
    LPImageGenerator.makeSequences(int(numImages), augmentedDir)
    print(numImages, " Augmented Images Generated")

    # Apply Augmentation
    for filename in os.listdir(augmentedDir):
        # Skip the csv file
        if "csv" in filename:
            continue

        # pick augmentation method
        if augmentationType == "GaussianNoise":
            apply_gaussian_noise(augmentedDir, filename)

    # Write TF Record
    writer = tf.io.TFRecordWriter("FromScratch/TFRecordAugmented.tfrecord")

    with open(augmentedDir+"/dataset.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            tf_example = TFRecordConverter.createTFRecord(row, augmentedDir)
            writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    main(sys.argv)
