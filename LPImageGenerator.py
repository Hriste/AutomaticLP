# Christina Paolicelli
# 01/25/2020
# Fuse of License Plate Sequence Generator & image generation

# Python Image Library - https://www.pythonware.com/products/pil/
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import numpy as np

import string
#from numpy.random import choice
from random import choice
import os
from datetime import datetime

def makeSequences(numOfPlates):
    # Capital letters - restricted to what is valid for Maryland Plates
    alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
    font = ImageFont.truetype("md.ttf",100)

    col = 375
    row = 192

    # Make a directory for the new images
    directoryName = datetime.now().strftime("GeneratedImages_%Y-%m-%d_%H-%M")
    try:
        os.mkdir(directoryName)
    except:
        print("Directory ", directoryName, " already exists.")

    # For each needed license plate generate a random sequence
    for i in range(numOfPlates):
        num1 = choice(range(10))
        num2 = choice(range(10))
        num3 = choice(range(10))
        num4 = choice(range(10))
        num5 = choice(range(10))
        a1 = choice(alpha)
        a2 = choice(alpha)

        # add the text to the image
        # Note: Image is modified in place
        img = Image.open("MDPlateStyle1.png")
        draw = ImageDraw.Draw(img)
        draw.text((round(.059*col), round(.24*row)),str(num1),(0,0,0),font=font)
        draw.text((round(.1902*col), round(.24*row)),a1,(0,0,0),font=font)
        draw.text((round(.3213*col), round(.24*row)),a2,(0,0,0),font=font)
        draw.text((round(.4525*col), round(.24*row)),str(num2),(0,0,0),font=font)
        draw.text((round(.5836*col), round(.24*row)),str(num3),(0,0,0),font=font)
        draw.text((round(.7148*col), round(.24*row)),str(num4),(0,0,0),font=font)
        draw.text((round(.8459*col), round(.24*row)),str(num5),(0,0,0),font=font)
        #draw = ImageDraw.Draw(img)

        # Make FileName (same as sequence)
        filename = str(num1)+a1+a2+str(num2)+str(num3)+str(num4)+str(num5)
        img.save(directoryName+"/"+filename+".png")

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

    sequences = makeSequences(int(numOfPlates));

if __name__ == '__main__':
    main()
