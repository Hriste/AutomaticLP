# Christina Paolicelli
# 01/25/2020
# Fuse of License Plate Sequence Generator & image generation

# Python Image Library - https://www.pythonware.com/products/pil/
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import string
from random import choice
from random import sample
import os
from datetime import datetime
import csv
from operator import itemgetter

import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_SHAPE = (128*4, 64*4)
FONT_HEIGHT = 32*4
FONT_SIZE = 24*4

#data_dict = {}
alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']


def makeSequences(numOfPlates, directoryName):
    # Capital letters - restricted to what is valid for Maryland Plates
    counts = [0]*32
    font = ImageFont.truetype("md.ttf",FONT_SIZE)

    col = 375
    row = 192

    # Make a directory for the new images
    try:
        os.mkdir(directoryName)
    except:
        print("Directory ", directoryName, " already exists.")

    print(directoryName)
    with open(directoryName+"/dataset.csv", 'w', newline='') as data_file:
        # use the below for matlab csv output
        #data = ["LP", "x1", "y1", "w1", "h1"]

        # Use the below for tensorflow / python
        #data = ["FileName", "Class", "X", "Y", "W", "H"]
        data = ["FileName", "Class", "Xmin", "Ymin", "Xmax", "Ymax" ]

        for entry in data:
            data_file.write(str(entry)+",")
        data_file.write("\n")


        # For each needed license plate generate a random sequence
        for i in range(numOfPlates):
            num1 = choice(range(10))
            num2 = choice(range(10))
            num3 = choice(range(10))
            num4 = choice(range(10))
            num5 = choice(range(10))
            # Use the below commented out if you need no duplicates
            #temp = sample(range(10), 5)
            #num1 = temp[0]
            #num2 = temp[1]
            #num3 = temp[2]
            #num4 = temp[3]
            #num5 = temp[4]
            a1 = choice(alpha)
            a2 = choice(alpha)
            #temp = sample(alpha, 2)
            #a1 = temp[0]
            #a2 = temp[1]

            # Use the below for the template
            # Note: Image is modified in place
            #img = Image.open("MDPlateStyle1.png")


            img = Image.new(mode = "RGB", size = OUTPUT_SHAPE, color = (255, 255, 255))
            draw = ImageDraw.Draw(img)

            # Calculate where to start the x location
            xloc = (OUTPUT_SHAPE[0]-(font.getsize(str(num1))[0] + font.getsize(str(num2))[0] +
            font.getsize(str(num3))[0] + font.getsize(str(num4))[0] +
            font.getsize(str(num5))[0] + font.getsize(a1)[0] +
            font.getsize(a2)[0]))/2

            # Make FileName (same as sequence)
            yloc = (OUTPUT_SHAPE[1] - font.getsize('a')[1])/2
            sequence = str(num1)+a1+a2+str(num2)+str(num3)+str(num4)+str(num5)
            draw.text((xloc, yloc), sequence, (0,0,0), font=font)

            #img.save(directoryName+"/"+sequence+".png")
            img.save(directoryName+"/"+sequence+".png")

            # Get corners of boxes for each char (x,y,w,h) - x, y are upper left
            xloc2 = xloc + font.getsize(str(num1))[0]
            xloc3 = xloc2 + font.getsize(a1)[0]
            xloc4 = xloc3 + font.getsize(a2)[0]
            xloc5 = xloc4 + font.getsize(str(num2))[0]
            xloc6 = xloc5 + font.getsize(str(num3))[0]
            xloc7 = xloc6 + font.getsize(str(num4))[0]

            # Use this for new tensorflow dataset
            filename = sequence+".png"
            data = [filename, zeroMap(sequence[0]), xloc, yloc, xloc+font.getsize(str(num1))[0], yloc+font.getsize(str(num1))[1],
                    alpha2num(sequence[1]), xloc2, yloc, xloc2+font.getsize(a1)[0], yloc+font.getsize(a1)[1],
                    alpha2num(sequence[2]), xloc3, yloc, xloc3+font.getsize(a2)[0], yloc+font.getsize(a2)[1],
                    zeroMap(sequence[3]), xloc4, yloc, xloc4+font.getsize(str(num2))[0], yloc+font.getsize(str(num2))[1],
                    zeroMap(sequence[4]), xloc5, yloc, xloc5+font.getsize(str(num3))[0], yloc+font.getsize(str(num3))[1],
                    zeroMap(sequence[5]), xloc6, yloc, xloc6+font.getsize(str(num4))[0], yloc+font.getsize(str(num4))[1],
                    zeroMap(sequence[6]), xloc7, yloc, xloc7+font.getsize(str(num5))[0], yloc+font.getsize(str(num5))[1]]
            for entry in data:
                data_file.write(str(entry)+",")
            data_file.write("\n")

            # increment counts
            counts[int(sequence[0])] += 1
            counts[alpha2num(sequence[1])] += 1
            counts[alpha2num(sequence[2])] += 1
            counts[int(sequence[3])] += 1
            counts[int(sequence[4])] += 1
            counts[int(sequence[5])] += 1
            counts[int(sequence[6])] += 1


        print("Number of Instance of Each Character:")
        for j in range(0,10):
            print(str(j) + " " + str(counts[j]))
        for k in range(0, 22):
            print(alpha[k]+" "+str(counts[k+10]))

def alpha2num(letter):
    # convert a letter class to a number (numbers are 0 - 9, so letters are 10 - 32)
    return alpha.index(letter) + 10;

def zeroMap(number):
    if number == 0 or number == "0":
        return 32
    else:
        return number

def main():
    while 1:
        numOfPlates = input("Enter the number of plate sequences / images needed: ")
        # Protect aganist invalid inputs
        if not numOfPlates.isnumeric():
            print("Please enter a numeric value \n")
        elif int(numOfPlates) <= 0:
            print("Please enter a Value above 0 \n")
        else:
            break
    directoryName = datetime.now().strftime("GeneratedImages_%Y-%m-%d_%H-%M")
    sequences = makeSequences(int(numOfPlates), directoryName);


if __name__ == '__main__':
    main()
