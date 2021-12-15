from os import listdir

import numpy as np
from time import time
import matplotlib.pyplot as plt
import pickle
import os
# import scipy.fft as fft
from scipy.fft import fft, ifft


def openPickleFile(openfile):
    with open(openfile, "rb") as file:
        obj = pickle.load(file)
        obj["filename"] = openfile[:-7]
    return obj


def plotData(obj):
    # read data from pickle obj
    recived_data = obj.get("received")
    sent_signal = obj.get("signal")
    recived_rate = obj.get("rate")
    recived_interval = obj.get("interval")
    recived_data_frequency = obj.get("data freq")
    recived_data_amp = obj.get("data amp")
    recived_offset = obj.get("tx_offset")
    time_line = obj.get("time line")
    filename = obj.get("filename")

    # plot received data #
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(time_line[300:400], recived_data[300:400])
    axs[1].plot(time_line[300:400], sent_signal[300:400])
    axs[1].set_xlabel(r"time $ \left[s\right] $")
    axs[0].set_ylabel(r"amplitude $ \left[V\right] $")
    axs[1].set_ylabel(r"amplitude $ \left[V\right] $")
    fig.suptitle("received signal")
    # fig.savefig(f"{filename}_recived.svg", bbox_inches='tight')
    fig.show()

    show_fft(recived_data, recived_interval, filename)

    # recived_data1 = recived_data  # todo
    # length_recived_data = len(recived_data1)
    #
    # ## plot fft of received data ##
    # # get fft of given function shifted
    # sig_fft = fft.fftshift(fft.fft(recived_data1))
    # # create freq vectors
    # frequencies = fft.fftshift(fft.fftfreq(length_recived_data, 1 / recived_rate))
    # plt.plot(frequencies, np.abs(sig_fft), "r")
    # # plt.plot(frequencies, np.imag(sig_fft), "g")
    # plt.title("received signal fft")
    # plt.xlabel("frequency[Hz]")
    # plt.ylabel("magnitude")
    # # plt.savefig(f"{filename}_fft_received.svg", bbox_inches='tight')
    # plt.show()


def show_fft(array: np.ndarray, duration: int = 1, filename="fft.svg"):
    """
    shows the fft of given function
    :param filename:
    :param array:
    :param duration:
    :return:
    """
    transform = fft(array)
    abs_transform = np.abs(transform[:transform.size // 2])
    frequencies = np.linspace(0, abs_transform.size / duration, abs_transform.size)
    # frequencies = np.linspace(0, abs_transform.size * 2 * np.pi / duration, abs_transform.size)
    plt.plot(frequencies, abs_transform)
    plt.yscale("log")
    plt.xlabel("Frequencies")
    plt.ylabel("magnitude")
    plt.title("received signal fft")
    plt.savefig(f"{filename}_fft_received.svg", bbox_inches='tight')
    plt.show()
    return frequencies, abs_transform


if __name__ == '__main__':
    # o = [openPickleFile(filename) for filename in os.listdir() if not (
    #         filename.endswith(".py") or filename.endswith(".jpg") or filename.endswith(".txt") or filename.endswith(
    #     "noise_cover_led.pickle") or filename.endswith(".svg"))]
    # for obj in o:
    #     plotData(obj)
    plotData(openPickleFile("nyquist_fr12111rate20000d0.02.pickle"))
    # for f in (f in listdir() if f.endswith(".pickle")):
    #     plotData(openPickleFile(f))
