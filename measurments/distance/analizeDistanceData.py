import numpy as np
from time import time
import matplotlib.pyplot as plt
import pickle
import os
from scipy.io import wavfile
from scipy.optimize import curve_fit

def openPickleFile(openfile):
    with open(openfile, "rb") as file:
        object = pickle.load(file)
    return object

def drawAllMeasurements(objectsLst):
    for obj in objectsLst:
        slicedData = obj.get("received")[10000:45000]
        plt.plot(np.arange(len(slicedData)) / obj.get("rate"), slicedData)

    plt.xlabel("Sec")
    plt.ylabel("Amplitude")
    plt.show()
    # plt.savefig(f"distanceAllMeasures.jpg")

def analizeDistEffect(objectsLst):
    amp = []
    dist = []
    for obj in objectsLst:
        slicedData = obj.get("received")[10000:45000] #sliced received data
        amp.append((max(slicedData) - min(slicedData))/2) #amplitude
        dist.append(obj.get("distance")) #distance

    plt.plot(dist,amp)
    plt.xlabel("distance [CM]")
    plt.ylabel("Amplitude [V]")
    plt.show()
    return dist,amp

def OnePartsTSquared(t, a, b, c):
    t = t[t + b != 0]
    res = a * (1 / ((t + b) ** 2)) + c
    return res

def OnePartsTSquared1(t: float, a, b, c):
    if t!= 0:
        res = a * (1 / ((t + b) ** 2)) + c
        return res
    return 0

##

def test():
    xdata = np.linspace(1, 4, 50)
    y = OnePartsTSquared(xdata, 1, 0, 0)
    rng = np.random.default_rng(2)
    y_noise = 0.2 * rng.normal(size=xdata.size)
    ydata = y + y_noise
    plt.plot(xdata, y)
    plt.plot(xdata, ydata, 'b-', label='data')

    popt, pcov = curve_fit(OnePartsTSquared, xdata, ydata)
    plt.plot(xdata, OnePartsTSquared(xdata, *popt))
    plt.show()
    # popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
    # plt.plot(xdata, func(xdata, *popt), 'g--',
    #          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

##



if __name__ == '__main__':
    o = [openPickleFile(filename) for filename in os.listdir() if not
    (filename.endswith(".py") or filename.endswith(".jpg") or filename.endswith(".txt"))]
    o.sort(key=(lambda x: x["distance"]))
    drawAllMeasurements(o)
    dist,amp = analizeDistEffect(o)
    np.savetxt('test1.txt', amp)  # x,y,z equal sized 1D arrays
    # demoFunction(dist,amp)
    # test()






