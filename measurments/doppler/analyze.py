import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.fft as fft

# plt.rcParams['text.usetex'] = True

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
    lim = np.digitize([39_680, 39_900], freq) - 1
    plt.plot(freq[lim[0]:lim[1]], np.abs(f_domain[lim[0]:lim[1]]))


SPEEDS = [0, 0.0367, 0.1064, -0.1064, -0.0367]
DV = [0, 4e-4, 1e-3, 1e-3, 1e-4]


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        if "speed" in data:
            data["dv"] = DV[data['speed']]
            data['speed'] = SPEEDS[data['speed']]
    return data


SAMPLE_COUNT = 250_000


def analyze3(data):
    data.sort(key=lambda x: x.get('speed'))
    ref = next(d for d in data if d.get("speed") == 0)
    freq = fft.fftfreq(SAMPLE_COUNT, 1 / ref['rate'])
    lim = np.digitize([39_900, 39_950], freq[:freq.size // 2]) - 1
    print("calculating fourier transform")
    for i, d in enumerate(data):
        print(i)
        d['f_domain'] = np.abs(fft.fft(d['received'][-2 * SAMPLE_COUNT:-SAMPLE_COUNT]))
        d['f_domain_sliced'] = d['f_domain'][lim[0]:lim[1]]
    print("finish calculating FT")
    print("calculating correlation")
    for i, d in enumerate(data):
        print(i)
        d['correlation'] = np.correlate(ref['f_domain_sliced'], d['f_domain_sliced'], 'same')
        # d['correlation'] = d['correlation'][d['correlation'].size//2-lim[2]:d['correlation'].size//2+lim[2]]
        d['shift'] = np.argmax(d['correlation']) - d['correlation'].size // 2  # - lim[2]
        if d['speed'] == SPEEDS[2]:
            f = plt.figure()
            plt.plot(freq[lim[0]:lim[1]], ref['f_domain_sliced'])
            plt.plot(freq[lim[0]:lim[1]], d['f_domain_sliced'])
            # plt.arrow(freq[np.argmax(d['f_domain_sliced'])], d['f_domain_sliced'].max(), d['shift']*freq[1], 0)
            plt.xlabel(r"f $ \left[Hz\right] $")
            plt.savefig("correlation.svg")
            plt.close(f)
    print("finish calculating correlation")
    x = [d['speed'] for d in data]
    dx = [d['dv'] for d in data]
    y = [freq[-d['shift']] for d in data]
    dy = 0.3 * freq[1]
    plt.errorbar(x, y, dy, dx, ".")
    plt.xlabel(r"Velocity $ \left[\frac{m}{s}\right] $")
    plt.ylabel(r"$\Delta f \left[Hz\right] $")


def plot(x, y, name, title, xlabel, ylabel):
    plt.figure(name + " " + title if title else name)
    plt.title(title)
    plt.xlabel(xlabel, )
    plt.ylabel(ylabel)
    plt.plot(x, y)


if __name__ == '__main__':
    data = [load_data(filename) for filename in os.listdir() if
            filename.endswith(".pickle") and not filename == "noise.pickle" and "take2" not in filename]
    print("starting with transmitter")
    analyze3([d for d in data if "moving part" in d and d['moving part'] == 'tr'])
    plt.title("transmitter moving")
    v = np.linspace(-0.11, 0.11, 200)
    plt.plot(v, 40_000 * v / (v + 343))
    plt.savefig("transmitter.svg")
    plt.show()
    print("finish with transmitter")
    # print("starting with receiver")
    # analyze3([d for d in data if "moving part" in d and (d['moving part'] == 'rec' or d['speed'] == 0)])
    # plt.title("receiver moving")
    # plt.savefig("receiver.svg")
    # plt.show()
    # print("finish with receiver")
    # for f in ["doppler_s-1mov'rec'_take1.pickle", "doppler_s0movtr.pickle", "doppler_s1mov'rec'_take1.pickle",
    #           "doppler_s2mov'rec'_take1.pickle"]:
    #     analyze2(f)
    # plt.xlabel(r"$ f \left[Hz\right] $")
    # plt.yscale("log")
    # plt.legend(["-1", "0", "1", "2"])
    # plt.show()
    # analyze(listdir()[2])
    # for file in (file for file in os.listdir() if file.endswith(".pickle") and not file == "noise.pickle"):
    #     analyze(file)
    #     break
    # f_domain = [analyze(file) for file in os.listdir() if
    #             file.endswith(".pickle") and file not in ["noise.pickle", "save_data.pickle"]]
    # with open("save_data.pickle", "wb+") as f:
    #     pickle.dump(f_domain, f)
