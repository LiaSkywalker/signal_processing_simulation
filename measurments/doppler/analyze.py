import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.fft as fft

plt.rcParams['text.usetex'] = True

SHOW = False


def figure_out(name_base, title):
    if SHOW:
        plt.show()
    else:
        plt.savefig(name_base + " " + title + ".svg")


def analyze(filename):
    name_base = filename.split(".")[0]
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    plot(np.arange(0, data['received'].size / data['rate'], 1 / data['rate']), data['received'], name_base,
         "received signal", r"time $ \left[s\right] $", "")
    figure_out(name_base, "received signal")

    f_domain = np.abs(fft.fft(data['received'])[:data['received'].size // 2])
    f_domain[0] = 0
    freq = fft.fftfreq(data['received'].size, 1 / data['rate'])[:data['received'].size // 2]

    plot(freq, f_domain, name_base, "frequency domain", r"$ f \left[Hz\right] $",
         "")
    plt.yscale("log")
    figure_out(name_base, "frequency domain")

    idx = np.argmax(f_domain)
    lim = np.digitize([37_000, 43_000], freq) - 1
    plot(freq[lim[0]:lim[1]], np.abs(f_domain[lim[0]:lim[1]]), name_base, "frequency domain zoom",
         r"$ f \left[Hz\right] $", "")
    plt.yscale("log")
    # plt.xlim(39_000, 41_000)
    # plt.axvline(39_800, color='k')
    plt.annotate("$$ f=%.1f $$" % freq[idx], (freq[idx], np.abs(f_domain[idx])))
    figure_out(name_base, "frequency domain zoom")
    with open("results2.txt", "a+") as f:
        if "speed" in data:
            part = "transmitter" if data["moving part"] == 'tr' else "receiver"
            f.write("%s\tspeed=%2d" % (part, data["speed"]))
        else:
            f.write(name_base + " \t")
        f.write("\tfrequency=%.3f\tdF=%.3f\n" % (freq[idx], freq[1]))

    idx = np.argmax(data['received'][data['received'].size // 4:]) + (data['received'].size // 4)
    sliced_data = data['received'][idx - 5000:idx + 5000]

    plot(np.linspace(0, sliced_data.size / data['rate'], sliced_data.size), sliced_data, name_base,
         "received signal sliced", r"time $ \left[s\right] $", "")
    figure_out(name_base, "received signal sliced")

    f_domain2 = np.abs(fft.fft(sliced_data)[:sliced_data.size // 2])
    f_domain2[0] = 0
    freq2 = fft.fftfreq(sliced_data.size, 1 / data['rate'])[:sliced_data.size // 2]
    idx = np.argmax(f_domain2)

    plot(freq2, f_domain2, name_base, "frequency domain zoom (sliced)",
         r"$ f \left[Hz\right] $", "")
    plt.yscale("log")
    # plt.xlim(39000, 41000)
    # plt.axvline(39800, color='k')
    plt.annotate("$$ f=%.1f $$" % freq2[idx], (freq2[idx], f_domain2[idx]))
    figure_out(name_base, "frequency domain zoom (sliced)")
    plt.close('all')
    return {"freq": freq2, "f_domain": f_domain2, "speed": data["speed"] if "speed" in data else 0,
            'moving part': data['moving part'] if 'moving part' in data else "NONE"}


def analyze2(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    f_domain = np.abs(fft.fft(data['received'])[:data['received'].size // 2])
    f_domain[0] = 0
    freq = fft.fftfreq(data['received'].size, 1 / data['rate'])[:data['received'].size // 2]
    idx = np.argmax(f_domain)
    lim = np.digitize([39_600, 40_100], freq) - 1
    plt.plot(freq[lim[0]:lim[1]], np.abs(f_domain[lim[0]:lim[1]]))



def plot(x, y, name, title, xlabel, ylabel):
    plt.figure(name + " " + title if title else name)
    plt.title(title)
    plt.xlabel(xlabel, )
    plt.ylabel(ylabel)
    plt.plot(x, y)


if __name__ == '__main__':
    for f in ["doppler_s-1mov'tr'_take1.pickle", "doppler_s0movtr.pickle", "doppler_s1mov'tr'_take1.pickle", "doppler_s2mov'tr'_take1.pickle"]:
        analyze2(f)
    plt.xlabel(r"$ f \left[Hz\right] $")
    plt.yscale("log")
    plt.legend(["-1","0","1","2"])
    plt.show()
    # analyze(listdir()[2])
    # for file in (file for file in os.listdir() if file.endswith(".pickle") and not file == "noise.pickle"):
    #     analyze(file)
    #     break
    # f_domain = [analyze(file) for file in os.listdir() if
    #             file.endswith(".pickle") and file not in ["noise.pickle", "save_data.pickle"]]
    # with open("save_data.pickle", "wb+") as f:
    #     pickle.dump(f_domain, f)
