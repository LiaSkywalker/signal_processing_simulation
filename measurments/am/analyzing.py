import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from pickle import load
from os import listdir


# plt.rcParams["text.usetex"] = True


def get_graphs(filename):
    """
    print all the graphs for AM measurement
    :param filename: path to pickle file constants the measurement data.
    """
    name_base = filename.split('.')[0]
    with open(filename, "rb") as f:
        measurement = load(f)

    rate = measurement["rate"]
    measurement["time line"] *= 1000  # convert to milliseconds

    plt.figure(name_base + " original signal")
    plt.title("original signal")
    plt.xlabel("time [ms]")
    plt.plot(measurement["time line"], measurement["signal"])
    plt.savefig(name_base + " original signal.svg")
    # plt.show()

    rx = measurement["received"]

    plt.figure(name_base + " received signal")
    plt.title("received signal")
    plt.xlabel("time [ms]")
    plt.plot(measurement["time line"], rx)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    # plt.savefig(name_base + " received signal.svg")
    # plt.show()

    rx = rx[round(rx.size * 0.6):]
    time_line = measurement["time line"]
    time_line = time_line[round(time_line.size * 0.6):]

    plt.figure(name_base + "received signal sliced")
    plt.title("received signal sliced")
    plt.xlabel("time [ms]")
    plt.plot(time_line, rx)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    plt.savefig(name_base + " received signal sliced.svg")
    # plt.show()

    plt.figure(name_base + "received signal zoom")
    plt.title("received signal zoom")
    plt.xlabel("time [ms]")
    plt.plot(time_line[:round(rate * 2 / measurement["data freq"])], rx[:round(rate * 2 / measurement["data freq"])])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    # plt.savefig(name_base + " received signal zoom.svg")
    # plt.show()

    rx_fft = fft.fftshift(fft.fft(rx))
    freq = fft.fftshift(fft.fftfreq(rx.size, 1 / rate))
    plt.figure(name_base + "AM frequency domain")
    plt.plot(freq, np.abs(rx_fft))
    plt.title("AM frequency domain")
    plt.xlabel(r"frequency $\left[Hz\right]$")
    plt.ylabel(r"magnitude")
    plt.yscale("log")
    plt.savefig(name_base + " AM frequency domain.svg")
    # plt.show()

    idx = np.digitize([measurement["carrier freq"] + i * measurement["data freq"] for i in [-2, 2]], freq) - 1
    zoom_abs = np.abs(rx_fft[idx[0]:idx[1]])

    plt.figure(name_base + "AM frequency domain zoom")
    plt.plot(freq[idx[0]:idx[1]], zoom_abs)
    plt.axvline(measurement["carrier freq"], color="k")
    plt.title("AM frequency domain zoom")
    plt.xlabel(r"frequency $\left[Hz\right]$")
    plt.ylabel(r"magnitude")
    plt.yscale("log")
    plt.savefig(name_base + " AM frequency domain zoom.svg")
    # plt.show()

    data = demodulate_am(2 * rx / (rx.max() - rx.min()), rate, measurement["carrier freq"],
                         measurement["carrier freq"] / 3)

    plt.figure(name_base + "demodulated data")
    plt.title("demodulated data")
    plt.xlabel("time [ms]")
    plt.plot(time_line, data)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    plt.savefig(name_base + " demodulated data.svg")
    # plt.show()

    plt.figure(name_base + "demodulated data zoom")
    plt.title("demodulated data zoom")
    plt.xlabel("time [ms]")
    plt.plot(time_line[:round(rate / measurement["data freq"])], data[:round(rate / measurement["data freq"])])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    # plt.savefig(name_base + " demodulated data zoom.svg")
    # plt.show()


def demodulate_am(signal: np.ndarray, rate: float, carrier_freq: float, band_radius: float = 40_000) -> np.ndarray:
    """
    demodulate amplitude modulation.
    :param signal: the modulated signal.
    :param rate: sample rate of the signal.
    :param carrier_freq: The frequency of the carrier.
    :param band_radius: The frequency limit of the modulated data.
    :return: the data of the signal.
    """
    t = np.linspace(0, signal.size / rate, signal.size)
    print(f"{len(t)=}")
    carrier = np.cos(t * carrier_freq * 2 * np.pi)
    return band_pass(signal * carrier, rate, 20, band_radius) * 4


def band_pass(signal: np.ndarray, rate: float, low: float, high: float):
    """
    filter out all the frequencies that outside the given band.
    :param signal: signal to filter.
    :param rate: the sample rate of the signal.
    :param low: the lower bound of the frequencies to pass.
    :param high: the upper bound of the frequencies to pass.
    :return: filtered signal.
    """
    sig_fft = fft.fft(signal)
    freq = fft.fftfreq(sig_fft.size, 1 / rate)
    abs_freq = np.abs(freq)

    # plt.figure("pre")  # todo: delete
    # plt.title("pre")
    # plt.xlabel("f [Hz]")
    # plt.plot(fft.fftshift(freq), fft.fftshift(sig_fft))
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    # plt.show()

    sig_fft[abs_freq > high] = 0
    sig_fft[abs_freq < low] = 0

    # plt.figure("post")  # todo: delete
    # plt.title("post")
    # plt.xlabel("f [Hz]")
    # plt.plot(fft.fftshift(freq), fft.fftshift(sig_fft))
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    # plt.show()

    return fft.ifft(sig_fft)


if __name__ == '__main__':
    for file in (file for file in listdir() if file.endswith(".pickle")):
        get_graphs(file)
