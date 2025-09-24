from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy import signal as sig
from sys import argv


@dataclass
class dataset:
    short: np.array
    long: np.array
    time: np.array
    channel: np.array

    def subset(self, condition: np.array):
        return dataset(self.short[condition], self.long[condition], self.time[condition], self.channel[condition])

    def y(self):
        return (self.long - self.short) / self.long


def load_dataset(filename):
    raw = np.transpose(np.loadtxt(argv[1], delimiter=','))
    return dataset(short=raw[1], long=raw[0], time=raw[2], channel=raw[3])


show_n_gamma = True

def find_range(y, bins, cutoff):
    current_range = [min(y), max(y)]
    h, _ = np.histogram(y, bins, range=current_range)

    while h[0] < cutoff * max(h):
        bin_dist = (current_range[1] - current_range[0]) / bins
        current_range[0] += 0.5 * bin_dist
        h, _ = np.histogram(y, bins, range=current_range)

    while h[-1] < cutoff * max(h):
        bin_dist = (current_range[1] - current_range[0]) / bins
        current_range[1] -= 0.5 * bin_dist
        h, _ = np.histogram(y, bins, range=current_range)

    return current_range


def gaussian(x, a, mu, sigma_sq):
    return a * np.exp(- ((x - mu) ** 2) / (2 * sigma_sq))


def double_gaussian(x, a_1, a_2, mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return gaussian(x, a_1, mu_1, sigma_sq_1) + gaussian(x, a_2, mu_2, sigma_sq_2)


def analyze_run(data):
    data = data.subset(data.long > data.short)
    y = data.y()

    autorange = find_range(y, 100, 0.005)
    counts, bins = np.histogram(y, bins=100, range=autorange)
    bin_centers = bins[0:-1] + 0.5 * (bins[1] - bins[0])

    peaks = sig.argrelmax(counts)[0]

    biggest, second = 0, 0

    for peak in peaks:
        if counts[peak] > counts[biggest]:
            biggest = peak

    for peak in peaks:
        if counts[peak] > counts[second] and np.abs(peak - biggest) > 20:
            second = peak

    peak_two = bin_centers[biggest]
    peak_one = bin_centers[second]

    sigma_guess_one = 1.7e-2
    sigma_guess_two = 1.7e-2

    amp_one = counts[second]
    amp_two = counts[biggest]

    params, covariance = opt.curve_fit(
        double_gaussian, bin_centers, counts,
        p0=[amp_one, amp_two, peak_one, peak_two, sigma_guess_one, sigma_guess_two],
        maxfev=99999
    )

    errors = np.sqrt(np.diag(covariance))

    sigma_gamma = np.sqrt(params[4])
    sigma_neutron = np.sqrt(params[5])
    peak_diff = np.abs(params[3] - params[2])

    fom = peak_diff / (2.355 * (sigma_gamma + sigma_neutron))
    print("FOM = ", fom)

    mu_gamma = params[2]
    mu_neutron = params[3]

    x_values = np.linspace(bins[0], bins[-1], 1000)
    fitted_curve = double_gaussian(x_values, *params)
    between_peaks = (x_values > mu_gamma) & (x_values < mu_neutron)
    cutoff_value = x_values[fitted_curve == min(fitted_curve[between_peaks])][0]

    if show_n_gamma:
        print("found values are:")
        print(f"cutoff value = {cutoff_value}")
        print(f"sigma_gamma_sq = {params[4]} +- {errors[4]}")
        print(f"sigma_neutron_sq = {params[5]} +- {errors[5]}")
        print(f"mu_gamma = {params[2]} +- {errors[2]}")
        print(f"mu_neutron = {params[3]} +- {errors[3]}")
        print(f"a_gamma = {params[0]} +- {errors[0]}")
        print(f"a_neutron = {params[1]} +- {errors[1]}")

        plt.stairs(counts, bins)
        plt.plot(x_values, double_gaussian(x_values, *params))
        plt.vlines(cutoff_value, 0, max(fitted_curve) * 0.8)
        plt.ylabel("count")
        plt.xlabel(r"$\frac{long - short}{long}$")
        plt.show()

    return cutoff_value


def csv_print(fname, *cols):
    with open(fname, 'w') as file:
        file.writelines([", ".join([str(c[i]) for c in cols]) + "\n" for i in range(len(cols[0]))])


def pairs(iterable):
    iterable = iter(iterable)
    while True:
        try:
            a = next(iterable)
            b = next(iterable)
            yield a, b
        except StopIteration:
            return None


def main():
    data = load_dataset(argv[1])

    for i in range(1, len(data.time)):
        if data.time[i] < data.time[i - 1]:
            data.time[i:] += 4294967296

    timing_offset = 0 # todo: make this the actual number for the offset from pulse to motion

    # run n-gamma-discrimination
    sync_channel = 3
    data_channel = 1
    cutoff = 0.3 # analyze_run(data.subset(data.channel == data_channel))
    timing_pulses = data.subset(data.channel == sync_channel).time
    data = data.subset(data.channel == data_channel)
    print(f"number of timing pulses is: {len(timing_pulses)}")
    neutron_hits = data.subset(data.short < data.long)
    neutron_hits = neutron_hits.subset(neutron_hits.y() > cutoff)

    # convert hits to fluency
    time_const = 6.25e-8 # assumes 12.5ns
    times = neutron_hits.time


    # assosicate position values to data
    fluencies = []
    x_points = []
    y_points = []
    line_count = len(timing_pulses) // 2
    current_line = 0
    fwd = True
    for start, end in pairs(timing_pulses):
        start -= timing_offset
        end -= timing_offset
        delta_t = end - start
        timeborders = np.linspace(start, end, line_count + 1)
        for i in range(0, line_count):
            mask = (times >= timeborders[i]) & (times <= timeborders[i + 1])
            fluencies.append(len(times[mask]) / delta_t)
            x_points.append(i if fwd else line_count - i)
            y_points.append(current_line)
        current_line += 1

    # write result
    csv_print(argv[2], x_points, y_points, fluencies)


if __name__ == "__main__":
    main()
