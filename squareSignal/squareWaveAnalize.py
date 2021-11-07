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
    f = 80

    ## plot received data ##
    plt.plot(recived_data[3900:5000])
    plt.ylabel("amplitude [V]")
    plt.title("received square signal")
    plt.savefig(f"{filename}_recived.svg", bbox_inches='tight')
    plt.show()

    ## plot original data ##
    plt.plot(original_signal[3900:5000])
    plt.ylabel("amplitude [V]")
    plt.title("transmitted square signal")
    plt.savefig(f"{filename}_original.svg", bbox_inches='tight')
    plt.show()

    recived_data1 = recived_data[3900:5000]  # todo
    length_recived_data = len(recived_data1)

    ## plot fft of received data ##
    # get fft of given function shifted
    sig_fft = fft.fftshift(fft.fft(recived_data1))
    # create freq vectors
    frequencies = fft.fftshift(fft.fftfreq(length_recived_data, 1 / recived_rate))
    idx = np.digitize([f + i for i in [-f, f]], frequencies) - 1
    zoom_abs = np.abs(sig_fft[idx[0]:idx[1]])
    zoom_frequencies = np.abs(frequencies[idx[0]:idx[1]])

    plt.plot(zoom_frequencies, zoom_abs, "r")
    # plt.plot(frequencies, np.imag(sig_fft), "g")
    plt.title("received signal fft")
    plt.xlabel("frequency[Hz]")
    plt.ylabel("magnitude")
    plt.yscale("log")

    plt.savefig(f"{filename}_fft_received.svg", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    o = [openPickleFile(filename) for filename in os.listdir() if not (
            filename.endswith(".py") or filename.endswith(".jpg") or filename.endswith(".txt") or filename.endswith(
        "noise_cover_led.pickle") or filename.endswith(".svg"))]
    for obj in o:
        plotData(obj)
