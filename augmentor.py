import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import csv
import imageio
import random

WHITE_LIST_FORMAT = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'JPG')
VALID_AUGMENTERS = ["jpegCompression", "MotionBlur", "Affine", "All", "Gaussian"]

HEIGHT = 256
WIDTH = 512

def select_augmentation():
    print("Valid Augmentation Values are:")
    for entry in VALID_AUGMENTERS:
        print(entry)

    choosen = input("Please select what augmenation you would like to perform:")

    if "jpegcompression" in choosen.lower():
        level = int(input("Please enter a severity level between 1-5, or random:"))
        augmentor = iaa.imgcorruptlike.JpegCompression(severity=level)
    elif "motionblur" in choosen.lower():
        kernel = int(input("Enter the motion blur kernel size in pixels (suggestion 15):"))
        # TODO: can add angle(s) for motion blur
        augmentor = iaa.MotionBlur(k=kernel)
    elif "GaussianNoise" in choosen.lower():
        level = int(input("Please enter a severity level between 1-5, or random:"))
        augmentor = iaa.imgcorruptlike.GaussianNoise(severity=level)
    elif "affine" in choosen.lower():
        # TODO: there's alot more affine options
        # I decided to implement it this way could implemet other ways
        print("/nYou're going to be asked to enter a number of Affine parameters:")
        angleMin = int(input("Please enter the min roatation angle in degrees (suggestion -15): "))
        angleMax = int(input("Please enter the max rotation angle in degrees (suggestion 15): "))
        yMin = float(input("Please enter the min % to scale on the y axis (suggestion 0.5 (50%)): "))
        yMax = float(input("Please enter the max % to scale on the y axis (suggestion 1.5 (150%)): "))
        xMin = float(input("Please enter the min % to scale on the x axis (suggestion 0.5 (50%)): "))
        xMax = float(input("Please enter the max % to scale on the x axis (suggestion 1.5 (150%)): "))
        augmentor = iaa.Affine(scale={"x":(xMin, xMax), "y":(yMin, yMax)}, rotate=(angleMin, angleMax))
        #augmentor = iaa.Sequential([iaa.Affine(rotate=(0, angleMax))])
    elif "all" in choosen.lower():
        augmentor = None
    else:
        augmentor = None

    return augmentor, choosen

def updateBoundingBoxes(row, augmentor, image):
    ia.seed(random.randrange(1,100,1))
    #image = ia.quokka(size=(WIDTH, HEIGHT))
    # Bounding box coordinates start at row[2]
    # and go in the order Xmin, Ymin, Xmax, Ymax
    # there are 7 bounding boxes per image & a class label between each one
    bbs = BoundingBoxesOnImage([
    BoundingBox(x1 = float(row[2]), y1=float(row[3]), x2=float(row[4]), y2=float(row[5])),
    BoundingBox(x1 = float(row[7]), y1=float(row[8]), x2=float(row[9]), y2=float(row[10])),
    BoundingBox(x1 = float(row[12]), y1=float(row[13]), x2=float(row[14]), y2=float(row[15])),
    BoundingBox(x1 = float(row[17]), y1=float(row[18]), x2=float(row[19]), y2=float(row[20])),
    BoundingBox(x1 = float(row[22]), y1=float(row[23]), x2=float(row[24]), y2=float(row[25])),
    BoundingBox(x1 = float(row[27]), y1=float(row[28]), x2=float(row[29]), y2=float(row[30])),
    BoundingBox(x1 = float(row[32]), y1=float(row[33]), x2=float(row[34]), y2=float(row[35]))
    ], shape = image.shape)

    image_aug, bbs_aug = augmentor(image = image, bounding_boxes=bbs)

    # update new_row with new BB Values
    offset = 2
    #modified = bbs_aug.remove_out_of_image().clip_out_of_image()
    #print(len(bbs.bounding_boxes))
    for i in range(len(bbs_aug.bounding_boxes)):
        #print(offset, offset+3)
        current = bbs_aug.bounding_boxes[i]
        row[offset] = str(current.x1)
        row[offset+1] = str(current.y1)
        row[offset+2] = str(current.x2)
        row[offset+3] = str(current.y2)
        offset = offset + 5

    # For debug
    #image_aug = bbs_aug.draw_on_image(image_aug, size=2, color=[0,0,255])
    #imshow(image_aug)
    return image_aug, row

def useAllAugmentations(path):
    rowsToAdd = []
    with open(path + "/dataset.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            new_row = row
            image_name = row[0]
            new_name = image_name.split(".")

            # Apply JPEG Compression
            jpeg_augmentor = iaa.imgcorruptlike.JpegCompression(severity=random.randrange(1,5))
            jpeg_name = new_name[0] + "_jpegCompression.jpg"
            new_row[0] = jpeg_name
            image = imageio.imread(path + "/" + image_name)
            ia.seed(1)
            image_aug = jpeg_augmentor(image = image)
            rowsToAdd.append(new_row)
            im_name = path + "/" + jpeg_name
            imageio.imwrite(im_name, image_aug[:, :, 0])

            # Apply GaussianNoise
            gaussian_augmentor = iaa.imgcorruptlike.GaussianNoise(severity=random.randrange(1,5))
            gaussian_name = new_name[0] + "_gaussian.jpg"
            new_row[0] = gaussian_name
            image = imageio.imread(path + "/" + image_name)
            ia.seed(1)
            image_aug = gaussian_augmentor(image = image)
            rowsToAdd.append(new_row)
            im_name = path + "/" + gaussian_name
            imageio.imwrite(im_name, image_aug[:, :, 0])

            # Motion blur
            mb_augmentaor = iaa.MotionBlur(k=random.randrange(4,25))
            mb_name = new_name[0] + "_motionBlur.jpeg"
            new_row = row
            new_row[0] = mb_name
            image = imageio.imread(path + "/" + image_name)
            ia.seed(1)
            image_aug = mb_augmentaor(image = image)
            rowsToAdd.append(new_row)
            im_name = path + "/" + mb_name
            imageio.imwrite(im_name, image_aug[:, :, 0])

            # Affine
            affine_augmentor = iaa.Affine(scale={"x":(0.5, 1.5), "y":(0.5, 1.5)}, rotate=(-5, 5))
            affine_name = new_name[0] + "_affine.jpeg"
            new_row = row
            new_row[0] = affine_name
            image = imageio.imread(path + "/" + image_name)
            image_aug, modified_row = updateBoundingBoxes(new_row, affine_augmentor, image)
            rowsToAdd.append(modified_row)
            im_name = path + "/" + affine_name
            imageio.imwrite(im_name, image_aug[:, :, 0])

        with open(path + "/dataset.csv", 'a') as fd:
            for row in rowsToAdd:
                stringToWrite = ""
                for entry in row:
                    stringToWrite = stringToWrite + entry + ","
                fd.write(stringToWrite)
                fd.write("\n")

def augment_images(path):
    rowsToAdd = []

    augmentor, choosen = select_augmentation()
    print(choosen)
    if (augmentor is None) and ("all" not in choosen.lower()):
        print("No Augmentation Type selected")
        return
    if "all" not in choosen.lower():
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

                if "affine" in choosen.lower():
                    # some augmentors also need the bounding boxes updated
                    image_aug, modified_row = updateBoundingBoxes(new_row, augmentor, image)
                    rowsToAdd.append(modified_row)
                else:
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
    else:
        useAllAugmentations(path)
