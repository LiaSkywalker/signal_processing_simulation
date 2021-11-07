import numpy as np
from time import time
import matplotlib.pyplot as plt
import pickle
import os
import scipy.fft as fft


def openPickleFile(openfile):
    with open(openfile, "rb") as file:
        object = pickle.load(file)
        object["filename"] = openfile
    return object


def plotData(obj):
    # read data from pickle obj
    recived_data = obj.get("received")
    original_signal = obj.get("signal")
    recived_rate = obj.get("rate")
    recived_interval = obj.get("interval")
    filename = obj.get("filename")[:-7]

    ## plot received data ##
    plt.plot(recived_data[29000:30000])
    plt.ylabel("amplitude [V]")
    # plt.xlabel("time []")
    plt.title("received triangle signal")
    plt.savefig(f"{filename}_recived.svg", bbox_inches='tight')
    plt.show()

    ## plot original data ##
    plt.plot(original_signal[29000:30000])
    plt.ylabel("amplitude [V]")
    # plt.xlabel("time []")
    plt.title("transmitted triangle signal")
    plt.savefig(f"{filename}_original.svg", bbox_inches='tight')
    plt.show()

    recived_data1= recived_data[20000:30000] #todo
    length_recived_data = len(recived_data1)

    ## plot fft of received data ##
    # get fft of given function shifted
    sig_fft = fft.fftshift(fft.fft(recived_data1))
    # create freq vectors
    frequencies = fft.fftshift(fft.fftfreq(length_recived_data, 1 / recived_rate))
    plt.plot(frequencies, np.abs(sig_fft), "r")
    # plt.plot(frequencies, np.imag(sig_fft), "g")
    plt.title("received signal fft")
    plt.xlabel("frequency[Hz]")
    plt.ylabel("magnitude")
    plt.savefig(f"{filename}_fft_received.svg", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    o = [openPickleFile(filename) for filename in os.listdir() if not (
                filename.endswith(".py") or filename.endswith(".jpg") or filename.endswith(".txt") or filename.endswith(
            "noise_cover_led.pickle") or filename.endswith(".svg"))]
    for obj in o:
        plotData(obj)