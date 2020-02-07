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

OUTPUT_SHAPE = (128*4, 128*4)#64*4)
FONT_HEIGHT = 32*4

data_dict = {}

def makeSequences(numOfPlates, directoryName):
    # Capital letters - restricted to what is valid for Maryland Plates
    alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
    font = ImageFont.truetype("md.ttf",24*4)

    col = 375
    row = 192

    # Make a directory for the new images
    try:
        os.mkdir(directoryName)
    except:
        print("Directory ", directoryName, " already exists.")

    print(directoryName)
    with open(directoryName+"/dataset.csv", 'w', newline='') as data_file:
        #writer = csv.writer(data_file, delimiter=",")
        #writer.writerow(["LP", "x1", "y1", "w1", "h1"])
        data = ["LP", "x1", "y1", "w1", "h1"]
        for entry in data:
            data_file.write(str(entry)+",")
        data_file.write("\n")

        # For each needed license plate generate a random sequence
        for i in range(numOfPlates):
            #num1 = choice(range(10))
            #num2 = choice(range(10))
            #num3 = choice(range(10))
            #num4 = choice(range(10))
            #num5 = choice(range(10))
            temp = sample(range(10), 5)
            num1 = temp[0]
            num2 = temp[1]
            num3 = temp[2]
            num4 = temp[3]
            num5 = temp[4]
            #a1 = choice(alpha)
            #a2 = choice(alpha)
            temp = sample(alpha, 2)
            a1 = temp[0]
            a2 = temp[1]

            # add the text to the image
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

            data = [sequence, xloc, yloc, font.getsize(str(num1))[0], font.getsize(str(num1))[1],
                                    xloc2, yloc, font.getsize(a1)[0], font.getsize(a1)[1],
                                    xloc3, yloc, font.getsize(a2)[0], font.getsize(a2)[1],
                                    xloc4, yloc, font.getsize(str(num2))[0], font.getsize(str(num2))[1],
                                    xloc5, yloc, font.getsize(str(num3))[0], font.getsize(str(num3))[1],
                                    xloc6, yloc, font.getsize(str(num4))[0], font.getsize(str(num4))[1],
                                    xloc7, yloc, font.getsize(str(num5))[0], font.getsize(str(num5))[1]]
            for entry in data:
                data_file.write(str(entry)+",")
            data_file.write("\n")



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

    # Sort CSV alpheticanly to match order in folder

    #with open(directoryName+"/dataset.csv", mode='r') as f:
        #data = [line for line in csv.reader(f)]

    #data.sort(key=itemgetter(1))

    #with open(directoryName+"/dataset.csv", mode='w') as f:
        #csv.writer(f).writerow(data)

if __name__ == '__main__':
    main()
