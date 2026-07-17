import data_loading
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sys import argv


def edge_fn(x, a, x0, sigma, b, c):
    return a * sp.special.erfc((x - x0) / (np.sqrt(2) * sigma)) + b * x + c


def local_mins(data, bins):
    data = np.log(data)
    data[data < 0] = 0
    data = sp.signal.medfilt(data, 25)
    data = np.convolve(data, np.ones(25), "same")
    diff = np.gradient(data, bins)
    plt.plot(bins, diff)
    # plt.plot(bins, data)
    plt.show()


def main():
    fname = argv[1]
    data = data_loading.load_dataset(fname)
    print(data.long)
    hist, bins = np.histogram(data.long, bins=1500)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    x0_guesses = local_mins(hist, bin_centers)
    plt.plot(bin_centers, hist)
    plt.yscale("log")
    plt.show()



if __name__ == "__main__":
    main()
