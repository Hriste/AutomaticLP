import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

FILENAME = 'trainingLog.txt'

def parseOutputFile():
    with open(FILENAME, 'r') as f:
        lines = f.readlines()
        f.close()

    steps = []
    loss = []
    secondsPerStep = []

    for line in lines:
        # If it's not a step info line - skip the processing
        if "INFO:tensorflow:global step" not in line:
            continue

        parts = line.split()
        #print(parts[5])
        steps.append(int(parts[2].replace(":","")))
        loss.append(float(parts[5]))
        secondsPerStep.append(float(parts[6].replace("(", "")))

    #print(steps)
    #print(loss)
    #print(secondsPerStep)

    # use %matplotlib inline
    fig, ax1 = plt.subplots()
    ax1.plot(steps, loss)
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Global Steps v. Loss")
    plt.show()

    #fileName =  datetime.now().strftime("Loss_%Y-%m-%d_%H-%M")
    #plt.savefig(fileName)

    fig2, ax2 = plt.subplots()
    ax2.plot(steps, secondsPerStep)
    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("Seconds Per Step")
    ax2.set_title("Seconds Per Step")
    plt.show()




parseOutputFile()
