#!python3
from matplotlib import pyplot as plt
from sys import argv
import numpy as np
import scipy as sp
from progress_print import pbar
import inspect
import sciebo_fetch

sampleset = 3
basepath = ""

if sampleset == 1:
    base_path = "../../Szintipulse/Cyl_Scint/Cyl_Scint/"
elif sampleset == 2:
    base_path = "../../Szintipulse/CylPMT_1400V/"
elif sampleset == 3:
    base_path = "../../Szintipulse/SquPMT_860V/"


def path_n(n: int) -> str:
    n = str(n)
    if len(n) < 5:
        n = "0" * (5 - len(n)) + n
    if sampleset == 1:
        return base_path + f"C4--Trace--{n}.txt"
    elif sampleset == 2:
        return base_path + f"C2--CylPMT_1400V--{n}.txt"
    elif sampleset == 3:
        return base_path + f"C2--SquPMT_860V--{n}.txt"


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


if __name__ == "__main__":
    plot_hist = False
    only_avg = False
    if len(argv) > 4 and argv[4] == "hist":
        plot_hist = True
    if len(argv) > 4 and argv[4] == "avg":
        only_avg = True

    ys = np.zeros(int(argv[2]) - int(argv[1]) + 1)
    long = np.zeros(int(argv[2]) - int(argv[1]) + 1)
    print(f"loading {len(ys)} files")
    rejected = 0
    n_gamma_cutoff = 0.45 # TODO
    start_file, stop_file = int(argv[1]), int(argv[2])
    progress_bar = pbar(stop_file - start_file, "", 40)
    cache = sciebo_fetch.open(argv[3])
    # temp = np.transpose(np.loadtxt(path_n(start_file), delimiter= "\t" if sampleset == 1 else ",", skiprows=5))
    # t, u = temp[0], temp[1]
    delim = "\t"
    t, u = None, None
    try:
        t, u = np.genfromtxt(cache.np_loadable(cache.ls()[start_file]), delimiter=delim, unpack=True, skip_header=5)
    except ValueError:
        delim = ","
        t, u = np.genfromtxt(cache.np_loadable(cache.ls()[start_file]), delimiter=delim, unpack=True, skip_header=5)
    t, u = retrigger(t, u)
    t, u = trim(t, u)
    avg_pulse_gamma = np.zeros(len(t))
    avg_pulse_neutron = np.zeros(len(t))
    refrence_timescale = t
    gamma_count, neutron_count = 0, 0
    i = -1
    for fname in cache.ls()[start_file:stop_file]:
        i += 1
        progress_bar.next()
        # fname = path_n(i)
        # data = np.transpose(np.loadtxt(fname, delimiter= "\t" if sampleset == 1 else ",", skiprows=5))
        # t, u = data[0], data[1]
        t, u = np.genfromtxt(cache.np_loadable(fname), delimiter=delim, unpack=True, skip_header=5)
        u -= np.average(u[:50])
        u = norm(u)
        if pile_up_reject(t, u):
            # print(f"file {i} got rejected for pile up reasons")
            rejected += 1
            # continue
        t, u = retrigger(t, u)
        t, u = trim(t, u)
        y, long_sample = psd(t, u)
        ys[i] = y
        long[i] = long_sample
        if not plot_hist and not only_avg:
            plt.plot(t, u)
        if only_avg:
            if y > n_gamma_cutoff:
                neutron_count += 1
                avg_pulse_neutron += (u / min(u))
            else:
                gamma_count += 1
                avg_pulse_gamma += (u / min(u))

            

    print(f"rejected {rejected} for pile ups")


    
    if plot_hist:
        print(ys)
        plt.hist2d(np.abs(long[ys != 0]), ys[ys != 0], (75, 75), range=[[0, 4e-8], [0.2, 0.75]])
        # plt.hist(ys[ys != 0], 50, (0, 1))
        # plt.hist2d(np.abs(long[ys != 0]), ys[ys != 0], (75, 75))
        # plt.hist(ys[ys != 0], 100, (0, 1) )
    elif only_avg:
        n_time, n_amp = retrigger(refrence_timescale, avg_pulse_neutron / neutron_count)
        gamma_time, gamma_amp = retrigger(refrence_timescale, avg_pulse_gamma / gamma_count)
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
