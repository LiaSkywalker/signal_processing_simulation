import nidaqmx
import nidaqmx.system
from nidaqmx.stream_readers import CounterReader
from nidaqmx.constants import (AcquisitionType, CountDirection, Edge,
                               READ_ALL_AVAILABLE, TaskMode, TriggerType)
import numpy as np
from time import time
import matplotlib.pyplot as plt
import pickle
from scipy.io import wavfile

#finding the connections of instruments
system = nidaqmx.system.System.local()
#select device
device = system.devices[1] if system.devices else None

def turn_off_led():
    """
    write zero voltage to device
    :return:
    """
    with nidaqmx.Task() as tx:
        tx.ao_channels.add_ao_voltage_chan(device.name + "/ao0")
        tx.write(0, True)


def take_measurements(signal: np.ndarray, sampling_rate: float,offset=3.):
    """

    :param signal:
    :param sampling_rate:
    :return:
    """
    duration = signal.size / sampling_rate
    with nidaqmx.Task() as rx, nidaqmx.Task() as tx, nidaqmx.Task() as clock:

        # analog input channel
        rx.ai_channels.add_ai_current_chan(device.name + "/ai0")
        rx.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)

        # analog output channel
        tx.ao_channels.add_ao_voltage_chan(device.name + "/ao0")
        tx.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)

        #clock channel?
        # clock.di_channels.add_di_chan(device.name+"/port0")
        # clock.timing.cfg_samp_clk_timing(sumpeling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        tx.stop() # make sure tx stopped
        tx.write(signal + offset) #write output signal

        #read data from rx
        vals = rx.read(signal.size)
        tx.stop()

        tx.write([0, 0, 0], True) # turn off diode
        tx.stop()

    return vals


def distance(x1):
    w = 570 #output frequency
    TimeInterval = 5 #time interval
    rate = 10000 #sampling rate
    x0 = 20 # initial location of diode on axis
    dis = x1 - x0 #
    offset = 3
    filename = f"distance/distance{dis}.pickle" #output filename
    signal = np.sin(w * np.linspace(0, TimeInterval, TimeInterval * rate) * 2 * np.pi) / 2
    values = take_measurements(signal, rate, offset)

    plt.plot(np.arange(values.size)/rate, values)
    plt.xlabel("Sec")
    plt.ylabel("Amplitude")
    plt.show()
    plt.savefig(f"distance/measureDistance{dis}.jpg")
    #save data to file
    with open(filename, "wb+") as f:
        pickle.dump({"offset_voltage": offset, "signal": signal, "received": values, "interval": TimeInterval,
                     "rate": rate, "time": time(), "distance": dis}, f)


def noise():
    w = 570
    T = 5
    rate = 10000
    # x0 = 20
    # dis = x1 - x0
    filename = f"noise_cover_photo_diode.pickle"
    sig = np.sin(w * np.linspace(0, T, T * rate) * 2 * np.pi) / 2
    values = take_measurements(sig, rate)
    plt.plot(values)
    plt.show()
    with open(filename, "wb+") as f:
        pickle.dump({"signal": sig, "received": values, "interval": T, "rate": rate,
                     "time": time(), "fileName": filename}, f)


def song(filename, output_name="out.wav"): #todo
    sample_rate, data = wavfile.read(filename)
    data = data[:, 0] / (2 ** 16)
    values = take_measurements(data[0:10 * sample_rate], sample_rate)
    wavfile.write(output_name, sample_rate, values)


# def read(f, normalized=False):
#     """MP3 to numpy array"""
#     a = pydub.AudioSegment.from_mp3(f)
#     y = np.array(a.get_array_of_samples())
#     if a.channels == 2:
#         y = y.reshape((-1, 2))
#     if normalized:
#         return a.frame_rate, np.float32(y) / 2**15
#     else:
#         return a.frame_rate, y
#
# def write(f, sr, x, normalized=False):
#     """numpy array to MP3"""
#     channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
#     if normalized:  # normalized array - each item should be a float in [-1, 1)
#         y = np.int16(x * 2 ** 15)
#     else:
#         y = np.int16(x)
#     song = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
#     song.export(f, format="mp3", bitrate="320k")

if __name__ == "__main__":
    # distance(564654654)
    song("Cat Ievan Polkka (320 kbps).wav", "cat320out.wav")
    # sample_rate, data = wavfile.read("Cat Ievan Polkka (320 kbps).wav", "cat320out.wav")
    # for dev in system.devices:
    #     print(f"{dev.name=}")
