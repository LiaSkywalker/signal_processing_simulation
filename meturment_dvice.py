import nidaqmx
import nidaqmx.system
from nidaqmx.constants import AcquisitionType
import numpy as np
from time import time, sleep
import matplotlib.pyplot as plt
import pickle
from scipy.io import wavfile
import sounddevice as sd
from simulation import simulate_rx, simulate_tx


def turn_off_led():
    with nidaqmx.Task() as tx:
        tx.ao_channels.add_ao_voltage_chan(device.name + "/ao0")
        tx.write(0, True)


def take_measurements(signal: np.ndarray, sampling_rate: float):
    if device not in system.devices:
        vals = simulate_rx(simulate_tx(signal))
        sleep(signal.size / sampling_rate)
        return vals
    with nidaqmx.Task() as rx, nidaqmx.Task() as tx:
        rx.ai_channels.add_ai_current_chan(device.name + "/ai0")
        rx.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        tx.ao_channels.add_ao_voltage_chan(device.name + "/ao0")
        tx.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        tx.stop()
        tx.write(signal + 3)
        tx.start()
        vals = np.array(rx.read(signal.size))
        tx.stop()
    return vals


def distance(x1):
    w = 570
    duration = 5
    rate = 10000
    x0 = 20
    dis = x1 - x0
    filename = f"distance/distance{dis}.pickle"
    sig = np.sin(w * np.linspace(0, duration, duration * rate) * 2 * np.pi) / 2
    values = take_measurements(sig, rate)
    plt.plot(values)
    plt.show()
    with open(filename, "wb+") as f:
        pickle.dump({"signal": sig, "received": values, "interval": duration, "rate": rate, "time": time(), "distance": dis},
                    f)


def noise():
    w = 570
    duration = 5
    rate = 10000
    filename = f"noise_cover_photo_diode.pickle"
    sig = np.sin(w * np.linspace(0, duration, duration * rate) * 2 * np.pi) / 2
    values = take_measurements(sig, rate)
    plt.plot(values)
    plt.show()
    with open(filename, "wb+") as f:
        pickle.dump({"signal": sig, "received": values, "interval": duration, "rate": rate, "time": time()}, f)


def song(filename, output_name="out.wav"):
    sample_rate, data = wavfile.read(filename)
    data = np.array(data[:, 0] / (2 ** 15 - 1))
    sliced_data = data[0:10 * sample_rate]
    values = take_measurements(sliced_data, sample_rate)
    values -= np.average(values)
    values *= (2 ** 15 - 1) / max(values.max(), -values.min())
    values = values.round().astype("int16")
    print(f"{values.size=}\n{sample_rate=}")
    wavfile.write(output_name, sample_rate, values)


def play_song(filename):
    sample_rate, data = wavfile.read(filename)
    chunk_size = sample_rate // 1000
    data = np.array(data[:, 0] / (2 ** 15 - 1))
    pos_range = iter(range(0, data.size, chunk_size))

    def callback(outdata: np.ndarray, frames: int, time, status) -> None:
        pos = next(pos_range)
        chunk = data[pos:pos + chunk_size + 1]
        values = take_measurements(chunk, sample_rate * 1.5)
        values -= np.average(values)
        outdata[:, 0] = values

    with sd.OutputStream(sample_rate, chunk_size, channels=1, dtype="float32", callback=callback):
        sd.sleep(1000 * data.size // sample_rate)


system = nidaqmx.system.System.local()
for dev in system.devices:
    print(dev)
# device = system.devices["a"]
device = system.devices[input("write the device name to use: ")]
