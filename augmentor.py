import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import csv
import imageio

WHITE_LIST_FORMAT = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'JPG')

def augment_images(path):
    rowsToAdd = []

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
            compression = iaa.imgcorruptlike.JpegCompression(severity=2)
            image_aug = compression(image = image)
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
