#!python3
from matplotlib import pyplot as plt
from sys import argv
import numpy as np
import scipy as sp
from progress_print import pbar

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


short_integration =  12.5e-9 * 2
# short_integration =  6.75e-9
long_integration = 12.5e-9 * 32
offset = 0 # -8e-9
trigger_level = 0.025


def pile_up_reject(time, voltage):
    cv_voltage = np.abs(np.convolve(voltage, np.ones(10) / 10, "same"))
    peaks, _ = sp.signal.find_peaks(cv_voltage, prominence=0.05, width=10)
    multiple_peaks = len(peaks) != 1
    wrong_baseline_start = np.any(cv_voltage[:50] > 1 * trigger_level)
    wrong_baseline_end = np.any(cv_voltage[-20:] > 1 * trigger_level)
    return multiple_peaks or wrong_baseline_start or wrong_baseline_end


def retrigger(time, voltage):
    dt = time[np.abs(np.convolve(voltage, np.ones(10) / 10, "same")) > trigger_level][0]
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
    if len(argv) > 3 and argv[3] == "hist":
        plot_hist = True
    if len(argv) > 3 and argv[3] == "avg":
        only_avg = True
    ys = np.zeros(int(argv[2]) - int(argv[1]) + 1)
    long = np.zeros(int(argv[2]) - int(argv[1]) + 1)
    print(f"loading {len(ys)} files")
    rejected = 0
    n_gamma_cutoff = 0.45 # TODO
    start_file, stop_file = int(argv[1]), int(argv[2])
    progress_bar = pbar(stop_file - start_file, "", 40)
    temp = np.transpose(np.loadtxt(path_n(start_file), delimiter= "\t" if sampleset == 1 else ",", skiprows=5))
    t, u = temp[0], temp[1]
    t, u = retrigger(t, u)
    t, u = trim(t, u)
    avg_pulse_gamma = np.zeros(len(t))
    avg_pulse_neutron = np.zeros(len(t))
    refrence_timescale = t
    gamma_count, neutron_count = 0, 0
    for i in range(start_file, stop_file + 1):
        progress_bar.next()
        fname = path_n(i)
        data = np.transpose(np.loadtxt(fname, delimiter= "\t" if sampleset == 1 else ",", skiprows=5))
        t, u = data[0], data[1]
        u = norm(u)
        if pile_up_reject(t, u):
            # print(f"file {i} got rejected for pile up reasons")
            rejected += 1
            continue
        t, u = retrigger(t, u)
        t, u = trim(t, u)
        y, long_sample = psd(t, u)
        ys[i - int(argv[1])] = y
        long[i - int(argv[1])] = long_sample
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
        # print(ys)
        plt.hist2d(np.abs(long[ys != 0]), ys[ys != 0], (75, 75), range=[[0, 4e-8], [0.2, 0.75]])
        # plt.hist(ys[ys != 0], 150, (0.2, 0.75))
        # plt.hist2d(np.abs(long[ys != 0]), ys[ys != 0], (75, 75))
        # plt.hist(ys[ys != 0], 100, (0, 1) )
    elif only_avg:
        n_time, n_amp = retrigger(refrence_timescale, avg_pulse_neutron / neutron_count)
        gamma_time, gamma_amp = retrigger(refrence_timescale, avg_pulse_gamma / gamma_count)
        plt.plot(gamma_time, gamma_amp, label="$\\gamma$")
        plt.plot(n_time, n_amp,  label="$n$")
    else:
        # plt.yscale("log")
        _, _, y1, y2 = plt.axis()
        plt.fill_between(np.linspace(offset, long_integration + offset, 1000), np.ones(1000) * y1, np.ones(1000) * y2, color="blue", alpha=0.2)
        plt.fill_between(np.linspace(offset, short_integration + offset, 1000), np.ones(1000) * y1, np.ones(1000) * y2, color="green", alpha=0.2)
    plt.show()
