import numpy as np, matplotlib.pyplot as plt, scipy.fft as fft
from pickle import load
from os import listdir

plt.rcParams['text.usetex'] = True

SHOW = False

def figure_out(name_base, title):
    if SHOW:
        plt.show()
    else:
        plt.savefig(name_base + " "+title+".svg")


def analyze(filename):
    name_base = filename.split(".")[0]
    with open(filename, 'rb') as f:
        data = load(f)
        
    
    plot(np.arange(0, data['received'].size/data['rate'], 1/data['rate']), data['received'], name_base, "received signal", r"time $ \left[s\right] $", "")
    figure_out(name_base, "received signal")
    
    f_domain = fft.fft(data['received'])
    f_domain[0] = 0
    freq = fft.fftfreq(f_domain.size, 1/data['rate'])
    
    plot(fft.fftshift(freq), fft.fftshift(np.abs(f_domain)), name_base, "frequency domain", r"$ f \left[Hz\right] $", "")
    figure_out(name_base, "frequency domain")
    
    idx_pos, idx_neg = np.argmax(f_domain[:len(f_domain)//2]), np.argmax(f_domain[len(f_domain)//2:])
    
    plot(fft.fftshift(freq), fft.fftshift(np.abs(f_domain)), name_base, "frequency domain zoom", r"$ f \left[Hz\right] $", "")
    plt.xlim(39000, 41000)
    plt.axvline(39800, color='k')
    figure_out(name_base, "frequency domain zoom")
    
    sliced_data = data['received'][data['received'].size//2:]
    idx = np.argmax(sliced_data)
    sliced_data = sliced_data[max(0,idx-5000):idx+5000]
    
    plot(np.linspace(0, sliced_data.size/data['rate'], sliced_data.size), sliced_data, name_base, "received signal sliced", r"time $ \left[s\right] $", "")
    figure_out(name_base, "received signal sliced")
    
    f_domain2 = fft.fft(sliced_data)
    f_domain2[0] = 0
    freq2 = fft.fftfreq(f_domain2.size, 1/data['rate'])
    
    plot(fft.fftshift(freq2), fft.fftshift(np.abs(f_domain2)), name_base, "frequency domain zoom (sliced)", r"$ f \left[Hz\right] $", "")
    plt.xlim(39000, 41000)
    plt.axvline(39800, color='k')
    figure_out(name_base, "frequency domain zoom (sliced)")



def plot(x,y,name, title, xlabel, ylabel):
    plt.figure(name + " "+title if title else name)
    plt.title(title)
    plt.xlabel(xlabel, )
    plt.ylabel(ylabel)
    plt.plot(x, y)


if __name__ == '__main__':
    #analyze(listdir()[2])
    for file in (file for file in listdir() if file.endswith(".pickle") and not file == "noise.pickle"):
        analyze(file)
