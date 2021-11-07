import numpy as np
from time import time
import matplotlib.pyplot as plt
import pickle
import os
import scipy.fft as fft

from matplotlib.ticker import LogFormatter, FormatStrFormatter, LogFormatterMathtext, FuncFormatter
from scipy.io import wavfile
from scipy.optimize import curve_fit


def openPickleFile(openfile):
    with open(openfile, "rb") as file:
        object = pickle.load(file)
    return object

def plotData(obj):

        #read data from pickle obj
        recived_data = obj.get("received")
        length_recived_data = len(recived_data)
        original_signal = obj.get("signal")
        recived_rate = obj.get("rate")
        recived_interval = obj.get("interval")

        #plot received data
        plt.plot(recived_data[6000:6300])
        # plt.ylabel("received")
        # plt.title("received cos signal")
        plt.savefig("cos_received.svg", bbox_inches='tight')
        plt.show()

        # plot original data
        plt.plot(original_signal[6000:6300])
        # plt.ylabel("signal")
        # plt.title("original cos signal")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        plt.savefig("cos_original.svg", bbox_inches='tight')
        plt.show()


        # plot fft of received data
        #get fft of given function shifted
        sig_fft = fft.fftshift(fft.fft(recived_data))
        #create freq vectors
        frequencies = fft.fftshift(fft.fftfreq(length_recived_data, 1/recived_rate))
        plt.plot(frequencies[4200:5800], np.abs(sig_fft)[4200:5_800], "r")
        # plt.plot(frequencies, np.imag(sig_fft), "g")
        plt.title("received signal fft")
        plt.xlabel("frequency[Hz]")
        plt.ylabel("magnitude")
        plt.yscale("log")
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        plt.savefig("fft_received_cos.svg", bbox_inches='tight')
        plt.show()

        plt.plot(frequencies[5550:5590], np.abs(sig_fft)[5550:5590], "r")
        plt.title("received signal fft")
        plt.xlabel("frequency[Hz]")
        plt.ylabel("magnitude")
        plt.yscale("log")
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        plt.savefig("fft_received_cos_zomm_in.svg", bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    o = [openPickleFile(filename) for filename in os.listdir() if not (filename.endswith(".py") or filename.endswith(".jpg") or filename.endswith(".txt") or filename.endswith("noise_cover_led.pickle") or filename.endswith(".svg"))]
    for obj in o:
        plotData(obj)