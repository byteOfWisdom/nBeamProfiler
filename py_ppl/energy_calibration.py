import data_loading
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sys import argv

# all assumed values are in here for tuning in a single place
class config:
    filter_len = 25
    smoothing_kernel = np.ones(25) / 2
    long_bins = 1500
    prominence = 0.1
    width = 10


def edge_fn(x, a, x0, sigma, b, c):
    return a * sp.special.erfc((x - x0) / (np.sqrt(2) * sigma)) + b * x + c


def edge_guess(data, bins):
    data = np.log(data)
    data[data < 0] = 0
    data = sp.signal.medfilt(data, config.filter_len)
    data = np.convolve(data, config.smoothing_kernel, "same")
    diff = np.gradient(data, bins)
    diff *= -1
    diff[diff < 0] = 0
    diff = np.convolve(diff, config.smoothing_kernel, "same")
    peaks, _ = sp.signal.find_peaks(diff, width=config.width, prominence=config.prominence)
    print(f"initial guesses where placed at {bins[peaks]}")
    # plt.plot(bins, diff)
    # plt.scatter(bins[peaks], diff[peaks])
    # plt.show()
    return bins(peaks)


def main():
    fname = argv[1]
    data = data_loading.load_dataset(fname)
    print(data.long)
    hist, bins = np.histogram(data.long, bins=config.long_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    x0_guesses = edge_guess(hist, bin_centers)
    plt.plot(bin_centers, hist)
    plt.yscale("log")
    plt.show()



if __name__ == "__main__":
    main()
