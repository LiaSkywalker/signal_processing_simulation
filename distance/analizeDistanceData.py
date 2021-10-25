import numpy as np
from time import time
import matplotlib.pyplot as plt
import pickle
import os
from scipy.io import wavfile

def openPickleFile(openfile):
    with open(openfile, "rb") as file:
        object = pickle.load(file)
    return object



if __name__ == '__main__':
    o = [openPickleFile(filename) for filename in os.listdir() if not
    (filename.endswith(".py") or filename.endswith(".jpg"))]
    o.sort(key=(lambda x: x["distance"]))
    Amp = []
    dist = []
    for obj in o:
        # print(obj.get("time"))
        slicedData = obj.get("received")[10000:45000]
        Amp.append((max(slicedData) - min(slicedData))/2) #amplitude
        dist.append(obj.get("distance"))
        plt.plot(np.arange(len(slicedData)) / obj.get("rate"), slicedData)

    plt.xlabel("Sec")
    plt.ylabel("Amplitude")
    plt.show()

    plt.plot(dist,Amp)
    plt.show()
    # plt.savefig(f"distanceAllMeasures.jpg")



