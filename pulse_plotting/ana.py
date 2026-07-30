#!python3
from matplotlib import pyplot as plt
from sys import argv
import numpy as np
import scipy as sp
from progress_print import pbar
import inspect
import sciebo_fetch


short_integration =  12.5e-9 * 1
# short_integration =  6.75e-9
long_integration = 12.5e-9 * 16
offset = 0 # -8e-9
trigger_level = 0.025


def none(x):
    return isinstance(x, type(None))


def some(x):
    return not none(x)


def goodness_of_fit(data, fit):
    rss = sum((data - fit) ** 2)
    tss = sum((data - np.average(data)) ** 2)
    return 1 - (rss / tss)


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


def pulse_function(t, amplitude, tau, t0,  sigma):
    dt = t - t0
    # b = ((dt / sigma) - (sigma / tau)) / np.sqrt(2)
    b = (tau * dt - sigma**2) / (np.sqrt(2) * sigma * tau)
    # return b
    # d = (sigma ** 2 + tau * t0) / (np.sqrt(2) * sigma * tau)
    c = amplitude * (1 + sp.special.erf(b)) / (2 * tau)
    return c * np.exp(((sigma**2) - (2 * tau * dt)) / (2 * tau ** 2))


def pile_up_reject(time, voltage):
    cv_voltage = (-1) * np.convolve(voltage, np.ones(10) / 10, "same")
    peaks, _ = sp.signal.find_peaks(cv_voltage, prominence=0.05, width=10)
    multiple_peaks = len(peaks) != 1
    wrong_baseline_start = np.any(cv_voltage[:50] > 1 * trigger_level)
    wrong_baseline_end = np.any(cv_voltage[-20:] > 1 * trigger_level)
    return multiple_peaks or wrong_baseline_start or wrong_baseline_end


def retrigger(time, voltage):
    dt = time[(-1) * np.convolve(voltage, np.ones(10) / 10, "same") > trigger_level][0]
    return time - dt, voltage


def trim(time, voltage):
    time_of_interest = time < long_integration * 1.25
    return time[time_of_interest], voltage[time_of_interest]


def norm(voltage):
    # return voltage / np.min(voltage)
    return voltage / 1


def psd(time, voltage):
    short_mask = np.where((time > offset) & (time < short_integration + offset))
    short_value = sp.integrate.simpson(voltage[short_mask], time[short_mask])
    # short_value = np.sum(voltage[short_mask])
    long_mask = np.where((time > offset) & (time < long_integration + offset))
    long_value = sp.integrate.simpson(voltage[long_mask], time[long_mask])
    # long_value = np.sum(voltage[long_mask])
    y = (long_value - short_value) / long_value
    return y, long_value


def pre_process(time, voltage):
    voltage -= np.average(voltage[:50])
    voltage = norm(voltage)
    if pile_up_reject(time, voltage):
        return True, time, voltage
    time, voltage = retrigger(time, voltage)
    time, voltage = trim(time, voltage)
    return False, time, voltage
    

def parse_args():
    plot_hist = False
    only_avg = False
    if len(argv) > 4 and argv[4] == "hist":
        plot_hist = True
    if len(argv) > 4 and argv[4] == "avg":
        only_avg = True
    return plot_hist, only_avg


class analysis:
    def __init__(self, filename, start_file, stop_file, show_pbar=True):
        # functions that can be overwritten for how to process data
        self.retrigger_func = retrigger
        self.psd_func = psd
        self.normalizing_func = norm
        self.trimming_func = trim

        self.n_gamma_cutoff = 0.45 # TODO
        self.t, self.u = None, None
        self.plot_all = False
        self.start_file, self.stop_file = start_file, stop_file
        self.file_count = stop_file - start_file + 1
        self.ys = np.zeros(self.file_count)
        self.long = np.zeros(self.file_count)
        print(f"loading {len(self.ys)} files")

        # init various counters
        self.rejected = 0
        if show_pbar:
            self.progress_bar = pbar(self.file_count, "", 40)
            self.has_pbar = True
        else:
            self.has_pbar = False

        self.cache = sciebo_fetch.open(filename)
        self.file_names = self.cache.ls()
        self.delimiter = "\t"
        self.current_filenum = -1
        self.gamma_count, self.neutron_count = 0, 0

    def init_reader(self):
        t, u = None, None
        try:
            t, u = np.genfromtxt(self.cache.np_loadable(self.cache.ls()[self.start_file]), delimiter=self.delimiter, unpack=True, skip_header=5)
        except ValueError:
            self.delimiter = ","
            t, u = np.genfromtxt(self.cache.np_loadable(self.cache.ls()[self.start_file]), delimiter=self.delimiter, unpack=True, skip_header=5)

        t, u = self.retrigger_func.__call__(t, u)
        t, u = self.trimming_func(t, u)
        self.avg_pulse_gamma = np.zeros(len(t))
        self.avg_pulse_neutron = np.zeros(len(t))
        self.refrence_timescale = t

    def load_next(self):
        self.current_filenum += 1
        file = self.file_names[self.current_filenum + self.start_file]
        self.t, self.u = np.genfromtxt(self.cache.np_loadable(file), delimiter=self.delimiter, unpack=True, skip_header=5)

        if self.has_pbar:
            self.progress_bar.next()
        return self.current_filenum < self.file_count

    def pre_process(self):
        self.u -= np.average(self.u[:50])
        self.u = self.normalizing_func(self.u)
        if pile_up_reject(self.t, self.u):
            self.rejected += 1
            return False
        self.t, self.u = self.retrigger_func(self.t, self.u)
        self.t, self.u = self.trimming_func(self.t, self.u)
        return True

    def psd(self):
        self.y_current, self.long_current = self.psd_func(self.t, self.u)
        self.ys[self.current_filenum] = self.y_current
        self.long[self.current_filenum] = self.long_current
        if self.y_current > self.n_gamma_cutoff:
            self.neutron_count += 1
            self.avg_pulse_neutron += (self.u / min(self.u))
        else:
            self.gamma_count += 1
            self.avg_pulse_gamma += (self.u / min(self.u))

    def process(self):
        # pre processing does pile up rejection, so only do the rest if that goes ok
        if self.pre_process():
            self.psd()
            if self.plot_all:
                plt.plot(self.t, self.u)
            
    def run(self):
        self.init_reader()
        while(self.load_next()):
            self.process()


if __name__ == "__main__":
    plot_hist, only_avg = parse_args()
    filename = argv[3]
    start_file, stop_file = int(argv[1]), int(argv[2])
    ana = analysis(filename, start_file, stop_file)
    ana.plot_all = not (plot_hist or only_avg)
    ana.run()

    print(f"rejected {ana.rejected} for pile ups")


    
    if plot_hist:
        print(ana.ys)
        plt.hist2d(np.abs(ana.long[ana.ys != 0]), ana.ys[ana.ys != 0], (75, 75), range=[[0, 4e-8], [0.2, 0.75]])
    elif only_avg:
        n_time, n_amp = ana.refrence_timescale, ana.avg_pulse_neutron / ana.neutron_count
        gamma_time, gamma_amp = ana.refrence_timescale, ana.avg_pulse_gamma / ana.gamma_count
        n_res, _ = curve_fit(pulse_function, n_time, n_amp, [max(n_amp), 5e-9, n_time[np.argmax(n_amp)], 1e-9])
        gamma_res, _ = curve_fit(pulse_function, gamma_time, gamma_amp, [max(gamma_amp), 5e-9, gamma_time[np.argmax(gamma_amp)], 1e-9])
        print(n_res)
        plt.plot(gamma_time, gamma_amp, label="$\\gamma$")
        plt.plot(n_time, n_amp,  label="$n$")
        plt.plot(n_time, pulse_function(n_time, *n_res), label="neutron fit")
        plt.plot(gamma_time, pulse_function(gamma_time, *gamma_res), label="neutron fit")
    else:
        # plt.yscale("log")
        _, _, y1, y2 = plt.axis()
        plt.fill_between(np.linspace(offset, long_integration + offset, 1000), np.ones(1000) * y1, np.ones(1000) * y2, color="blue", alpha=0.2)
        plt.fill_between(np.linspace(offset, short_integration + offset, 1000), np.ones(1000) * y1, np.ones(1000) * y2, color="green", alpha=0.2)
    plt.show()
