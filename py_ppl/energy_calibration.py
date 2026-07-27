import data_loading
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sys import argv
import inspect
# import std
import numba


error_bar_def = {"fmt": " ", "elinewidth": 0.75, "capsize": 2}


@numba.njit
def linear(x, a, b):
    return a * x + b


def none(x):
    return isinstance(x, type(None))


def some(x):
    return not none(x)


def goodness_of_fit(data, fit):
    rss = sum((data - fit) ** 2)
    tss = sum((data - np.average(data)) ** 2)
    return 1 - (rss / tss)


def plt_errorbar(xval, yval, xerr=None, yerr=None, label=None, marker=None, alpha=None):
    params = error_bar_def
    params["alpha"] = alpha if alpha else 0.5
    params["zorder"] = 5
    if some(marker):
        params["fmt"] = marker
    if none(xerr):
        params["fmt"] = marker if marker else "."
        params["markersize"] = 5

    plt.errorbar(xval, yval, xerr=xerr, yerr=yerr, label=label, **params)


def curve_fit(func, x_values, y_values, p0=None, maxfev=1000, y_errors=None):
    if some(y_errors):
        y_errors[y_errors == 0] = np.nan

    argc = len(str(inspect.signature(func)).split()[1:])
    if none(p0):
        p0 = np.ones(argc)

    func = np.vectorize(func)
    params_cf, cov = sp.optimize.curve_fit(func, x_values, y_values, sigma=y_errors, p0=p0, maxfev=maxfev, absolute_sigma=False)
    std_devs_cf = np.sqrt(np.diag(cov))
    goodness_cf = goodness_of_fit(y_values, func(x_values, *params_cf))
    return params_cf, (std_devs_cf, goodness_cf)


def plt_finish(xlabel, ylabel, save_to=False):
    plt.gcf().set_size_inches(16/1.75, 9/1.75)
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()

    plt.cla()


def plt_func(f, params=None, label=None, xrange=None, alpha=None):
    import numpy as np
    xmin, xmax, _, _ = plt.axis()
    if some(xrange) and some(xrange[0]):
        xmin = xrange[0]
    if some(xrange) and some(xrange[1]):
        xmax = xrange[1]
    x = np.linspace(xmin, xmax, 10000)
    y = f(x) if none(params) else f(x, *params) 
    alpha = alpha if alpha else 1
    plt.plot(x, y, label=label, alpha=alpha, zorder=10)


# all assumed values are in here for tuning in a single place
class config:
    filter_len = 25
    smoothing_kernel = np.ones(25) / 2
    long_bins = 1500
    prominence = 0.1
    width = 10


gamma_line_db = {
    "Na22": [511e3, 1274.537e3],
    "Cs137":[661.657e3],
    "AmBe": [44383]
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
    res, (err, rsq) = curve_fit(f, bin_centers[start:], hist[start:], p0=p0, maxfev=9999999)
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
        plt_func(f, self.res)
        plt.yscale("log")
        plt.ylim(bottom=0.9)
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
    res, _ = curve_fit(linear, lines, energies)
    plt_errorbar(lines, energies)
    plt_func(linear, res)
    plt_finish("Bin", "Energie / eV")


if __name__ == "__main__":
    main()
