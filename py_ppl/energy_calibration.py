import data_loading
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sys import argv
import std

# all assumed values are in here for tuning in a single place
class config:
    filter_len = 25
    smoothing_kernel = np.ones(25) / 2
    long_bins = 1500
    prominence = 0.1
    width = 10


gamma_line_db = {
    "Na22": [511e3, 1200e3] 
}


def edge_fn(x, a, x0, sigma, b, c):
    return a * sp.special.erfc((x - x0) / (np.sqrt(2) * sigma)) + b * x + c


def make_multi_edge(n):
    return lambda x, *args: sum([edge_fn(x, args[i], args[i + 1], args[i + 2], args[i + 3], args[i + 4]) for i in range(0, 5 * n, 5)])

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
    # print(f"initial guesses where placed at {bins[peaks]}")
    # plt.plot(bins, diff)
    # plt.scatter(bins[peaks], diff[peaks])
    # plt.show()
    return peaks


def fit_edges(bin_centers, hist):
    x0_guesses = edge_guess(hist, bin_centers)
    p0 = []
    for guess in x0_guesses:
        p0 += [hist[guess], bin_centers[guess], 1000, 0, 1]
    start = np.where(hist == max(hist))[0][0]
    f = np.vectorize(make_multi_edge(len(x0_guesses)))
    res, (err, rsq) = std.curve_fit(f, bin_centers[start:], hist[start:], p0=p0)
    return res, (err, rsq)


class dataset_analysis:
    def __init__(self, fname, isotope=""):
        data = data_loading.load_dataset(fname)
        print(data.long)
        self.hist, bins = np.histogram(data.long, bins=config.long_bins)
        self.bin_centers = 0.5 * (bins[:-1] + bins[1:])
        self.isotope = isotope

    def fit(self):
        self.res, (self.err, self.rsq) = fit_edges(self.bin_centers, self.hist)
        self.n = len(self.res) // 5
        self.compton_edges = []
        for i in range(self.n):
            self.compton_edges += [self.res[5 * i + 1]]

    def plot(self):
        print(self.res)
        print(self.n)
        plt.plot(self.bin_centers, self.hist)
        f = np.vectorize(make_multi_edge(self.n))
        std.default.plt_func(f, self.res)
        plt.yscale("log")
        plt.show()


# everything is eV
def compton_edge(gamma_energy):
    me_csq = (sp.constants.c ** 2) * sp.constants.electron_mass / sp.constants.electron_volt
    return 2 * gamma_energy ** 2 / (me_csq + 2 * gamma_energy)


def make_calibration_data(list_of_datasets):
    compton_energies = []
    compton_lines = []
    for ds in list_of_datasets:
        if ds.isotope not in gamma_line_db.keys():
            print("dataset without known isotope")
            continue

        for i in range(ds.n):
            compton_lines += [ds.compton_edges[i]]
            compton_energies += [compton_edge(gamma_line_db[ds.isotope][i])]

    return compton_lines, compton_energies


def main():
    # call with: python energy_calibration.py [file] [isotope] [file] [isotope] ...

    # fname = argv[1]
    # ana = dataset_analysis(fname, "Na22")
    # ana.fit()
    # # ana.plot()
    # print(ana.compton_edges)

    i = 1
    data = []
    while i + 1 < len(argv):
        fname = argv[i]
        elem = argv[i + 1]
        i += 2

        ana = dataset_analysis(fname, elem)
        ana.fit()
        ana.plot()
        data += [ana]

    lines, energies = make_calibration_data(data)
    res, _ = std.curve_fit(std.linear, lines, energies)
    std.default.plt_errorbar(lines, energies)
    std.default.plt_func(std.linear, res)
    std.default.plt_finish("Bin", "Energie / eV")


if __name__ == "__main__":
    main()
