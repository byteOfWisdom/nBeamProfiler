import data_loading
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib
import os
from sys import argv
import inspect
import numba
import scipy
import sciebo_fetch
from odrpack import odr_fit as odr_fit_raw
# from sigfig import round         #import an easy way of scientific rounding


def PGF_plots():
    # activate pgf-plotting
    matplotlib.use("pgf")
    # print('Using the ' + matplotlib.get_backend() + ' backend') #for debugging

    #update parameter for the right output font/size
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif", # use serif/main font for text elements
        # 'font.size': 10,        # change text size
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        'figure.autolayout': True,
        # "text.usetex": True,     # use inline math for ticks - THIS BREAKS THE EXPORT!
    })
    return 

# PGF_plots() #save as PGF otherwise PDF

def Save_Plot(path, title):
    #create directory if not existing
    os.makedirs(path, exist_ok=True)
    #check if PGF is used and save plot
    if matplotlib.get_backend() == 'pgf':
        plt.savefig(path + title +'.pgf', format='pgf')
        print('Plot saved as PGF')
    else:
        plt.savefig(path + title +'.pdf')
        print('Plot saved as PDF')

def Scriptpath(file):
    # path to where 'file' is
    scriptpath = str(os.path.abspath(os.path.dirname(file))) + '/'
    # print("Script path is:") #for debugging
    # print(scriptpath) #for debugging
    return scriptpath

#set path the where executing script is
scriptpath = Scriptpath(__file__)

error_bar_def = {"fmt": " ", "elinewidth": 0.75, "capsize": 2}


@numba.njit
def linear(x, a, b):
    return a * x + b

@numba.njit
def log(x, a, b, c):
    return a * np.log(b * x + 1) + c * x

def inverse_log(x, a, b, c):
    return (a * b * scipy.special.lambertw(c * np.exp(c / (a * b) + x / a) / (a * b)) - c) / (b * c) #thanks wolfram alpha!


def none(x):
    return isinstance(x, type(None))


def some(x):
    return not none(x)


def goodness_of_fit(data, fit):
    rss = sum((data - fit) ** 2)
    tss = sum((data - np.average(data)) ** 2)
    return 1 - (rss / tss)

def red_chi_squ(data,fitvalues, no_fitparameters):
    chi_squ = sum( (data - fitvalues) ** 2 / data) # we assume possion errors: error_data = sqrt(data), therefore error_data**2 = data
    datapoints = len(data)
    red_chi_squ = chi_squ / (datapoints - no_fitparameters)
    print("Red. Chi2 is: " +str(red_chi_squ))
    return red_chi_squ


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


def curve_fit(func, x_values, y_values, p0=None, maxfev=100000, y_errors=None, bounds=(-np.inf, np.inf), ftol=1e-8, xtol=1e-8, gtol=1e-8):
    if some(y_errors):
        y_errors[y_errors == 0] = np.nan

    argc = len(str(inspect.signature(func)).split()[1:])
    if none(p0):
        p0 = np.ones(argc)

    func = np.vectorize(func)
    params_cf, cov = sp.optimize.curve_fit(func, x_values, y_values, sigma=y_errors, p0=p0, maxfev=maxfev, absolute_sigma=True, bounds=bounds, ftol=ftol, xtol=xtol, gtol=gtol, verbose=0)
    std_devs_cf = np.sqrt(np.diag(cov))
    goodness_cf = goodness_of_fit(y_values, func(x_values, *params_cf))
    return params_cf, (std_devs_cf, goodness_cf)


def odr_fit(func, x_values, y_values, p0=None, maxfev=100000, y_errors=None, x_errors=None, bounds=None, ftol=1e-8, xtol=1e-8, gtol=1e-8):
    x_weight = np.ones_like(x_values, dtype=np.float64)
    y_weight = np.ones_like(y_values, dtype=np.float64)
    if some(x_errors):
        x_weight[x_errors != 0.] = x_errors[x_errors != 0.] ** (-2.)
        x_weight = x_weight / np.max(x_weight)
        x_weight[x_errors == 0.] = 1.
    if some(y_errors):
        y_weight[y_errors != 0.] = y_errors[y_errors != 0.] ** (-2.)
        y_weight = y_weight / np.max(y_weight)
        y_weight[y_errors == 0.] = 1.

    argc = len(str(inspect.signature(func)).split()[1:])
    
    if none(p0):
        p0 = np.ones(argc)
    else:
        argc = len(p0)

    def func_odr(t, B): return func(t, *B)
    odr_run = odr_fit_raw(func_odr, x_values, y_values, beta0=p0, weight_x=x_weight, weight_y=y_weight, bounds=bounds, maxit=maxfev)

    params_odr = odr_run.beta
    std_devs_odr = odr_run.sd_beta

    goodness_odr = goodness_of_fit(y_values, func(x_values, *params_odr))

    #remove 0's from array - i know this can be done more elegant but it's late o'clock and i need to get this done ~.~
    x_values = np.where(y_values>0, x_values, np.nan)
    x_values = x_values[~np.isnan(x_values)]
    y_values = np.where(y_values>0, y_values, np.nan)
    y_values = y_values[~np.isnan(y_values)]

    
    red_chi2 = red_chi_squ(y_values,func(x_values, *params_odr), argc)

    return params_odr, (std_devs_odr, goodness_odr), red_chi2


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
    # plt.plot(x, y, label=label, alpha=alpha, zorder=10, color='blue')
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


gamma_line_db = { #these are the gamma energies of the isotopes in eV
    "Na": [511e3, 1274.537e3],
    "Cs":[661.657e3],
    "AmBe": [4438e3]
}

initial_guess_db = {
    "Na": [4.1e3, 14.7e3],
    "Cs":[6.1e3],# 20e3],
    "AmBe": [36.05e3]
}


def edge_fn(x, a, x0, sigma, b, c): #x0 is the compton edge position
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
        p0 += [hist[guess_bin], bin_centers[guess_bin], 1000, 0, 1] # a, x0, sigma, b, c
    start = np.where(hist[1:] == max(hist[1:]))[0][0]
    f = np.vectorize(make_multi_edge(n))
    start = max(start, np.argmin(np.abs(bin_centers - (min(x0_guesses) - 5e3))))
    # stop = np.argmin(np.abs(bin_centers - (x0_guesses[-1]))) + 100
    stop = np.argmin(np.abs(bin_centers - (max(x0_guesses) + 5e3)))
    # res, (err, rsq) = curve_fit(f, bin_centers[start:stop], hist[start:stop], p0=p0, maxfev=9999999, y_errors=np.sqrt(hist[start:stop]), ftol=1e-8, xtol=1e-8, gtol=1e-8)
    # res, (err, rsq) = curve_fit(f, bin_centers[start:stop], hist[start:stop], p0=p0, maxfev=9999999, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    res, (err, rsq), red_chi2 = odr_fit(f, bin_centers[start:stop], hist[start:stop], p0=p0, maxfev=9999, y_errors=np.sqrt(hist[start:stop]))
    # for i in range(len(res)):
    #     print(round(res[i], err[i], sep='external_brackets'))
    print(res)
    return res, (err, rsq), (bin_centers[start], bin_centers[stop]), red_chi2


class dataset_analysis:
    def __init__(self, data, isotope=""):
        # data = data_loading.load_dataset(fname)
        # print(data.long)
        self.hist, bins = np.histogram(data.long, bins=config.long_bins)
        self.bin_centers = 0.5 * (bins[:-1] + bins[1:])
        self.isotope = isotope

    def fit(self):
        self.n = len(gamma_line_db[self.isotope])
        # x0_guesses = compton_edge(np.array(gamma_line_db[self.isotope])) * config.guess_factor
        x0_guesses = initial_guess_db[self.isotope]
        self.res, (self.err, self.rsq), self.xrange, self.red_chi2 = fit_edges(self.bin_centers, self.hist, self.n, x0_guesses)
        self.compton_edges = []
        self.edge_errors = []
        for i in range(self.n):
            self.compton_edges += [self.res[5 * i + 1]]
            self.edge_errors += [self.err[5 * i + 1]]

    def plot(self, defer_show=False):
        plt.plot(self.bin_centers, self.hist)
        f = np.vectorize(make_multi_edge(self.n))
        # plt_func(f, self.res, f"$R^2={round(self.rsq, 3)}$", self.xrange)
        plt_func(f, self.res, "$\\chi_{\\mathrm{red}}^{2}= $" +str(round(self.red_chi2,2)), self.xrange)
        plt.yscale("log")
        # plt.title(self.isotope)
        if not defer_show:
            plt.ylim(bottom=0.9)
            plt_finish("Long", "Energie / eV")


# everything is eV
def compton_edge(gamma_energy): # Calculate compton edge energy from gamma energy
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

    # url = "https://uni-bonn.sciebo.de/public.php/dav/files/qArdCtRpESZDtYk/?accept=zip" #url with all files
    url = "https://uni-bonn.sciebo.de/public.php/dav/files/QFjNaqbJALJHYTS/?accept=zip"   #url with one file per source
    if len(argv) > 1 and argv[1] != "reload":
        url = argv[1]
    calibration_data = sciebo_fetch.fetch(url, "gamma_calib_data", "reload" in argv)

    plot_single = "single" in argv

    data = []
    # print(calibration_data.ls())

    # plt.figure(figsize=(6, 3), dpi=500)

    for fname in calibration_data.ls():
        if fname == 'Szinti_Calib/Cs137_2.csv':
            continue
        elem = None
        for element in gamma_line_db.keys():
            if element in fname:
                elem = element
                break

        long, short, time, channel, _ = np.genfromtxt(calibration_data.np_loadable(fname), delimiter=",", unpack=True)

        ds = data_loading.dataset(short, long, time, channel)
        ds = ds.subset(ds.long > ds.short)
        ds = ds.subset(ds.long > 0)
        # plt.cla()
        # plt.hist(ds.y(), 100, (0, 0.75))
        # plt.show()
        # n gamma discrimination (but go for the gammas!!)
        ds = ds.subset(ds.y() < 0.4)

        

        ana = dataset_analysis(ds, elem)
        ana.fit()
        ana.plot(defer_show=not plot_single)
        data += [ana]

    if not plot_single:
        plt.ylim(bottom=0.9)
        # plt.xlim(right = 65000)

        #secondary x-axis in MeV
        #fitparamter from the next step of the calibration
        a = 3.89724561e+04  
        b = 3.63842683e-01     
        c = -1.00000000e+00

        def MeV2channel_log(x):
            return a * np.log(b * x + 1) + c * x
        
        def channel2MeV_log(x):
            return ((a * b * scipy.special.lambertw(c * np.exp(c / (a * b) + x / a) / (a * b)) - c) / (b * c)).real #thanks wolfram alpha!

        ax = plt.subplot(111)
        # secax = ax.secondary_xaxis('top', functions=(channel2MeV_log, MeV2channel_log))
        # secax.set_xlabel("$E$ / MeV")

        #dashed line to indicate the trigger-threshold. eyeballed to be at channel 2000 for now
        cutoff = round(channel2MeV_log(2000) * 1e3, 1)
        plt.vlines(x=2000, ymin=0, ymax=plt.ylim()[1],color='black', linestyle='--')
        plt.text(x=1200, y=plt.ylim()[1] * 0.10, s=str(cutoff) + ' keV', fontsize=10, rotation=90, color='black', ha='center', va='center', bbox=None)
        plt.text(x=18000, y=15, s='Na-22', fontsize=10, rotation=-45, color='black', ha='center', va='center', bbox=None)
        plt.text(x=10000, y=15, s='Cs-137', fontsize=10, rotation=-45, color='black', ha='center', va='center', bbox=None)
        plt.text(x=32000, y=150, s='AmBe', fontsize=10, rotation=0, color='black', ha='center', va='center', bbox=None)

        # plt_finish("long / channel", "counts")
        plt.gcf().set_size_inches(6, 3)
        plt.grid(which="major")
        plt.grid(which="minor", linestyle=":", linewidth=0.5)
        plt.gca().minorticks_on()
        plt.xlabel("$Q_{long}$ / channel")
        plt.ylabel("counts")
        plt.legend(loc="upper right")
        plt.tight_layout()
        Save_Plot(scriptpath + "plots/", "compton_edge_fits")
        plt.show()
        plt.cla()

    # plt.figure(figsize=(6, 3), dpi=500)
    lines, energies, line_err = make_calibration_data(data)

    # res, (_, rsq) = curve_fit(log, np.array(energies) / 1e6, lines, p0=[1,1,-0.1], bounds=[(0,0,-100) , (np.inf, np.inf, 10)])
    res, (_, rsq) = curve_fit(log, np.array(energies) / 1e6, lines, p0=[2, 2, -1], y_errors=np.abs(line_err), bounds=[(0, 0, -1), (np.inf, np.inf, 100)])
    print(res) # print fitparamter for energy calibration

    plt_errorbar(np.array(energies)/1e6, lines, yerr=line_err, marker='bo', alpha=1)
    plt.text(x=0.45, y=15000, s='Na-22', fontsize=10, rotation=40, color='black', ha='center', va='center', bbox=None)
    plt.axhline(y=14500, xmin=0.3/(abs(plt.xlim()[0])+abs(plt.xlim()[1])), xmax=0.40/(abs(plt.xlim()[0])+abs(plt.xlim()[1])), linewidth=1, color='black')
    plt.axvline(x=0.34, ymin=3900/(abs(plt.ylim()[0])+abs(plt.ylim()[1])), ymax=5000/(abs(plt.ylim()[0])+abs(plt.ylim()[1])), linewidth=1, color='black')
    plt.text(x=1.3, y=6000, s='Cs-137', fontsize=10, rotation=0, color='black', ha='center', va='center', bbox=None)
    plt.axhline(y=6200, xmin=0.3/(abs(plt.xlim()[0])+abs(plt.xlim()[1])), xmax=0.35/(abs(plt.xlim()[0])+abs(plt.xlim()[1])), linewidth=1, color='black')
    plt.text(x=4.3, y=42000, s='AmBe', fontsize=10, rotation=0, color='black', ha='center', va='center', bbox=None)
    plt.xlim(left=0)
    # plt.xlim(right=65000)
    plt.xlim(right=10)
    plt.ylim(bottom=0)
    plt.ylim(top=70000)
    plt_func(log, res, f"$R^2={round(rsq, 3)}$")
    # plt_finish("$E$ / MeV", "long / channel", )
    plt.gcf().set_size_inches(6, 3)
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel("$E$ / MeV")
    plt.ylabel("long / channel")
    plt.legend(loc="lower right")
    plt.tight_layout()
    Save_Plot(scriptpath + "plots/", "scinti_calibration")
    plt.show()


if __name__ == "__main__":
    main()
