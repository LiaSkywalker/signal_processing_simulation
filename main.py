from scipy.fft import fft, ifft
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt


def show_fft(array: np.ndarray):
    frequency = fft(array)
    plt.plot(np.abs(frequency)[:frequency.shape[0]]//2)
    plt.show()


def show_stft(array: np.ndarray, fs: float):
    f, t, Zxx = stft(array, fs, nperseg=1000)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def simulate_tx(array: np.ndarray):
    return array + np.random.normal(0, (array.max() - array.min())/100, array.size)


def simulate_rx(array: np.ndarray):
    return (array[:-1]+array[1:])/2


if __name__ == '__main__':
    T = 1
    dt = 1e-3
    w = 100
    t = np.arange(0, T, dt)
    s = np.sin(w * 2 * np.pi * t)
    tx = simulate_tx(s)
    rx = simulate_rx(tx)
    plt.plot(t, s)
    plt.title("original signal")
    plt.show()
    show_fft(s)
    # show_stft(s, 1 / dt)
    plt.plot(t, tx)
    plt.title("transmit signal")
    plt.show()
    show_fft(tx)
    plt.plot(t[1:], rx)
    plt.title("receive signal")
    plt.show()
    show_fft(rx)

