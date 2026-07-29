import data_loading
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sys import argv
import inspect
# import std
import numba
import sciebo_fetch


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
    smoothing_kernel = np.ones(25) / 25
    roll_len = 30
    long_bins = 1500
    prominence = 0.1
    width = 10
    min_height = 0.4
    min_spacing = 100
    guess_factor = 7000 / 1e6


gamma_line_db = {
    "Na": [511e3, 1274.537e3],
    "Cs":[661.657e3],
    "AmBe": [4438e3]
}

initial_guess_db = {
    "Na": [4.5e3, 15.0e3],
    "Cs":[9e3],# 20e3],
    "AmBe": [37e3]
}


def edge_fn(x, a, x0, sigma, b, c):
    return a * sp.special.erfc((x - x0) / (np.sqrt(2) * sigma)) + b * x + c


def make_multi_edge(n):
    return lambda x, *args: sum([edge_fn(x, args[i], args[i + 1], args[i + 2], args[i + 3], args[i + 4]) for i in range(0, 5 * n, 5)])

def edge_guess(data, bins):
    data = np.log(data)
    data[data < 0] = 0
    plt.plot(bins, data / max(data))
    data = sp.signal.medfilt(data, config.filter_len)
    data = np.convolve(data, config.smoothing_kernel, "same")
    diff = data - np.roll(data, config.roll_len)
    diff *= -1
    diff[diff < 0] = 0
    diff[bins < 2000] = 0
    diff = np.convolve(diff, config.smoothing_kernel, "same")
    diff = diff / max(diff)
    peaks, info = sp.signal.find_peaks(diff, width=config.width, prominence=config.prominence, height=config.min_height, distance=config.min_spacing)
    print(f"initial guesses where placed at {bins[peaks]}")
    plt.plot(bins, diff / max(diff))
    plt.scatter(bins[peaks], diff[peaks])
    plt.show()
    return peaks


def fit_edges(bin_centers, hist, n, x0_guesses):
    # x0_guesses = edge_guess(hist, bin_centers)
    # x0_guesses = x0_guesses[:n]
    # return [None], ([None], None)
    p0 = []
    for guess in x0_guesses:
        guess_bin = np.argmin(np.abs(bin_centers - guess))
        p0 += [hist[guess_bin], bin_centers[guess_bin], 1000, 0, 1]
    start = np.where(hist[1:] == max(hist[1:]))[0][0]
    f = np.vectorize(make_multi_edge(n))
    start = max(start, np.argmin(np.abs(bin_centers - (min(x0_guesses) - 5e3))))
    # stop = np.argmin(np.abs(bin_centers - (x0_guesses[-1]))) + 100
    stop = np.argmin(np.abs(bin_centers - (max(x0_guesses) + 5e3)))
    res, (err, rsq) = curve_fit(f, bin_centers[start:stop], hist[start:stop], p0=p0, maxfev=9999999)
    return res, (err, rsq), (bin_centers[start], bin_centers[stop])


class dataset_analysis:
    def __init__(self, data, isotope=""):
        # data = data_loading.load_dataset(fname)
        print(data.long)
        self.hist, bins = np.histogram(data.long, bins=config.long_bins)
        self.bin_centers = 0.5 * (bins[:-1] + bins[1:])
        self.isotope = isotope

    def fit(self):
        self.n = len(gamma_line_db[self.isotope])
        # x0_guesses = compton_edge(np.array(gamma_line_db[self.isotope])) * config.guess_factor
        x0_guesses = initial_guess_db[self.isotope]
        self.res, (self.err, self.rsq), self.xrange = fit_edges(self.bin_centers, self.hist, self.n, x0_guesses)
        self.compton_edges = []
        self.edge_errors = []
        for i in range(self.n):
            self.compton_edges += [self.res[5 * i + 1]]
            self.edge_errors += [self.err[5 * i + 1]]

    def plot(self):
        plt.plot(self.bin_centers, self.hist)
        f = np.vectorize(make_multi_edge(self.n))
        plt_func(f, self.res, f"$R^2={round(self.rsq, 3)}$", self.xrange)
        plt.yscale("log")
        plt.title(self.isotope)
        plt.ylim(bottom=0.9)
        plt_finish("Long", "Energie / eV")


# everything is eV
def compton_edge(gamma_energy):
    me_csq = (sp.constants.c ** 2) * sp.constants.electron_mass / sp.constants.electron_volt
    return 2 * gamma_energy ** 2 / (me_csq + 2 * gamma_energy)


def make_calibration_data(list_of_datasets):
    compton_energies = []
    compton_lines = []
    line_errors = []
    for ds in list_of_datasets:
        if ds.isotope not in gamma_line_db.keys():
            print("dataset without known isotope")
            print(ds.isotope)
            continue

        for i in range(ds.n):
            compton_lines += [ds.compton_edges[i]]
            line_errors += [ds.edge_errors[i]]
            compton_energies += [compton_edge(gamma_line_db[ds.isotope][i])]

    return compton_lines, compton_energies, line_errors


def main():
    # call with: python energy_calibration.py 

    url = "https://uni-bonn.sciebo.de/public.php/dav/files/qArdCtRpESZDtYk/?accept=zip"
    if len(argv) > 1:
        url = argv[1]
    calibration_data = sciebo_fetch.fetch(url, "gamma_calib_data")

    data = []
    for fname in calibration_data.ls():
        elem = None
        for element in gamma_line_db.keys():
            if element in fname:
                elem = element
                break

        long, short, time, channel, _ = np.genfromtxt(calibration_data.np_loadable(fname), delimiter=",", unpack=True)

        ds = data_loading.dataset(short, long, time, channel)

        ana = dataset_analysis(ds, elem)
        ana.fit()
        ana.plot()
        data += [ana]

    lines, energies, line_err = make_calibration_data(data)
    res, (_, rsq) = curve_fit(linear, lines, energies)
    plt_errorbar(lines, energies, yerr=line_err)
    plt_func(linear, res, f"$R^2={round(rsq, 3)}$")
    plt_finish("Bin", "Energie / eV")


if __name__ == "__main__":
    main()
