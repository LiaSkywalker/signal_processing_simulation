from scipy.fft import fft, ifft
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt

TOTAL_TIME = 1e-1
DT = 1e-3
OMEGA = 100


def show_fft(array: np.ndarray):
    """
    shows the fft of given function
    :param array:
    :return:
    """
    frequency = fft(array)
    plt.plot(np.abs(frequency)[:frequency.shape[0]//2])
    plt.show()


def show_stft(array: np.ndarray, fs: float):
    """
    shows the stft of given function
    :param array:
    :param fs:
    :return:
    """
    f, t, Zxx = stft(array, fs, nperseg=1000)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def drawPlot(time: np.ndarray, function: np.ndarray, graph_1_title, graph_1_xlabel,
             frequency: np.ndarray, graph_2_title, graph_2_xlabel, outputname):
    """
    create double figure
    :param time:
    :param function:
    :param graph_1_title:
    :param graph_1_xlabel:
    :param frequency:
    :param graph_2_title:
    :param graph_2_xlabel:
    :param outputname:
    :return:
    """

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # make figure of original signal
    ax[0].plot(time, function, linestyle='solid', linewidth=2)
    ax[0].legend(loc='upper left')
    ax[0].set_title(graph_1_title)
    ax[0].set_xlabel(graph_1_xlabel)

    # make figure of signal fft
    ax[1].plot(np.abs(frequency)[:frequency.shape[0] // 2], linestyle='solid', linewidth=2)
    ax[1].legend(loc='upper left')
    ax[1].set_title(graph_2_title)
    ax[1].set_xlabel(graph_2_xlabel)

    # show graph and save
    fig.show()
    fig.savefig(outputname)

def simulate_tx(array: np.ndarray):
    """
    simulate the signal with noise after transmitting
    :param array:
    :return:
    """
    return array + np.random.normal(0, (array.max() - array.min())/50, array.size)

def simulate_rx(array: np.ndarray):
    """
    simulate the signal with noise after receiving
    :param array:
    :return:
    """
    return (array[:-1]+array[1:])/2


if __name__ == '__main__':

    t = np.arange(0, TOTAL_TIME, DT)
    sin_func = np.sin(OMEGA * 2 * np.pi * t)
    sin_frequency = fft(sin_func)
    drawPlot(t, sin_func, "original signal", "Time",
             sin_frequency, "Frequency space", "Frequency",
             "firstGraph.jpg")

    tx = simulate_tx(sin_func)
    tx_frequency = fft(tx)

    show_stft(sin_func, 1 / DT)

    drawPlot(t, tx, "transmitted signal", "Time",
             tx_frequency, "Frequency space", "Frequency",
             "secondGraph.jpg")
    show_stft(tx, 1 / DT)

    rx = simulate_rx(tx)
    rx_frequency = fft(rx)
    drawPlot(t[1:], rx, "received signal", "Time",
             rx_frequency, "Frequency space", "Frequency",
             "thirdGraph.jpg")
