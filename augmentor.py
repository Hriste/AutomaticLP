import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import csv
import imageio

WHITE_LIST_FORMAT = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'JPG')
VALID_AUGMENTERS = ["jpegCompression", "MotionBlur"]

def select_augmentation():
    print("Valid Augmentation Values are:")
    for entry in VALID_AUGMENTERS:
        print(entry)

    choosen = input("Please select what augmenation you would like to perform:")

    if "jpegcompression" in choosen.lower():
        level = int(input("Please enter a severity level between 1-5:"))
        augmentor = iaa.imgcorruptlike.JpegCompression(severity=level)
    elif "motionblur" in choosen.lower():
        kernel = int(input("Enter the motion blur kernel size in pixels (suggestion 15):"))
        # TODO: can add angle(s) for motion blur
        augmentor = iaa.MotionBlur(k=kernel)
    else:
        augmentor = None

    return augmentor

def augment_images(path):
    rowsToAdd = []

    augmentor = select_augmentation()
    if augmentor is None:
        print("No Augmentation Type selected")
        return

    with open(path + "/dataset.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            new_row = row
            image_name = row[0]
            new_name = image_name.split(".")
            new_name = new_name[0] + "_aug.jpg"
            new_row[0] = new_name
            image = imageio.imread(path + "/" + image_name)

            rowsToAdd.append(new_row)

            ia.seed(1)
            image_aug = augmentor(image = image)
            im_name = path + "/" + new_name
            imageio.imwrite(im_name, image_aug[:, :, 0])

    # ok now append to the csv so they get written to the tf records file
    with open(path + "/dataset.csv", 'a') as fd:
        for row in rowsToAdd:
            stringToWrite = ""
            for entry in row:
                stringToWrite = stringToWrite + entry + ","
            fd.write(stringToWrite)
            fd.write("\n")
