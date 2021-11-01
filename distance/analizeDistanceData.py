import numpy as np
from time import time
import matplotlib.pyplot as plt
import pickle
import os

from matplotlib.ticker import LogFormatter, FormatStrFormatter, LogFormatterMathtext, FuncFormatter
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
        slicedData = obj.get("received")[10000:45000]  # sliced received data
        amp.append((max(slicedData) - min(slicedData)) / 2)  # amplitude
        dist.append(obj.get("distance"))  # distance

    plt.plot(dist, amp)
    plt.xlabel("distance [CM]")
    plt.ylabel("Amplitude [V]")
    plt.show()
    return dist, amp


def openNoisePickle():
    """
    :return: the average noise amplitude
    """
    obj = openPickleFile("noise_cover_led.pickle")
    objData = obj.get("received")  # sliced received data

    maxValue = np.max(objData)
    minValue = np.min(objData)
    amp = ((max(objData) - min(objData)) / 2)  # amplitude

    plt.plot(objData)
    plt.plot(np.repeat(maxValue, len(objData)))
    plt.plot(np.repeat(minValue, len(objData)))
    ampVec = np.repeat(amp, len(objData))
    plt.plot(ampVec)
    plt.show()

    return amp


##    def OnePartsTSquared1(t: float, a, b, c):
        # if t != 0:
            # res = a * (1 / ((t + b) ** 2)) + c
            # return res

def test(dist, amp):
    def OnePartsTSquared(t, a, b, c):
        # t = t[t + b != 0]
        res = a * (1 / ((t + b) ** 2)) + c
        return res

    #test on generated data
    xdata = np.linspace(1, 4, 50)
    y = xdata**-2#OnePartsTSquared(xdata, 1, 0, 0)
    rng = np.random.default_rng(2)
    y_noise = 0.2 * rng.normal(size=xdata.size)
    ydata = y + y_noise
    plt.plot(xdata, y)
    plt.plot(xdata, ydata, 'b-', label='data')

    popt, pcov = curve_fit(OnePartsTSquared, xdata, ydata)
    plt.plot(xdata, OnePartsTSquared(xdata, *popt))
    print("test on generated data:")
    print(pcov)
    print(f"a: {popt[0]}, b: {popt[1]}, c: {popt[2]}")
    plt.show()

def distFitting(dist, amp):
    def OnePartsTSquared(t, a, b, c):
        # t = t[t + b != 0]
        res = a * (1 / ((t + b) ** 2)) + c
        return res

    #curve fit on my data
    ax: plt.Axes = plt.axes()
    plt.scatter(dist, amp)
    popt1, pcov1 = curve_fit(OnePartsTSquared, dist, amp, check_finite=False)
    x_fit = np.linspace(min(dist), max(dist), len(dist)*10)
    plt.plot(x_fit, OnePartsTSquared(x_fit, *popt1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, y: str(x*1000)))
    plt.ylabel("amplitude[m] 10^-3")
    # plt.yscale("symlog", linthresh=max(amp)*2)
    print("real data:")
    print(pcov1)
    print(f"a: {popt1[0]}, b: {popt1[1]}, c: {popt1[2]}")
    plt.show()


if __name__ == '__main__':
    o = [openPickleFile(filename) for filename in os.listdir() if not (filename.endswith(".py") or filename.endswith(".jpg") or filename.endswith(".txt"))]
    o.sort(key=(lambda x: x["distance"]))
    # drawAllMeasurements(o)
    dist, amp = analizeDistEffect(o)

    # o = openPickleFile("noise_cover_led.pickle")
    # openNoisePickle()
    # print("hi")

    # np.savetxt('amp.txt', amp)  # x,y,z equal sized 1D arrays
    # np.savetxt('dist.txt', dist)  # x,y,z equal sized 1D arrays
    # demoFunction(dist,amp)
    distFitting(dist, amp)
