from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
import fileinput
from scipy import optimize as opt
from scipy import signal as sig
import progress_print


def closest_time(times: np.ndarray, point: float) -> float:
    diff = np.abs(times - point)
    return times[diff == np.min(diff)][0]


def fix_timing_pulses(timing_data):
    time_deltas = timing_data[1:] - timing_data[:-1]
    # time_deltas = np.sort(time_deltas)[:- int(len(time_deltas) * 0.3)] # filter out too long deltas due to missing events
    long_deltas = time_deltas[time_deltas > np.average(time_deltas)]
    long_deltas = np.sort(long_deltas)[:- int(len(long_deltas) * 0.3)]  # filter out too long deltas due to missing events
    short_deltas = time_deltas[time_deltas < np.average(time_deltas)]

    correct_short = np.average(short_deltas)
    correct_long = np.average(long_deltas)
    print(correct_short)
    print(correct_long)

    # next_is_short = False# the last movement will always be a short one

    # needs_long = np.append([False], time_deltas > 1.75 * correct_long)
    # for i in reversed(range(len(needs_long))):
    #     if needs_long[i]:
    #         timing_data = np.insert(timing_data, i + 1, timing_data[i] + correct_long)

    # needs_short = np.append([False], time_deltas > 1.75 * correct_short)
    # for i in reversed(range(len(needs_short))):
    #     if needs_short[i]:
    #         timing_data = np.insert(timing_data, i + 1, timing_data[i] + correct_short)

    i = 0
    next_long = True
    insertions = 0
    while i < len(timing_data) - 1:
        delta = np.abs(timing_data[i] - timing_data[i + 1])
        if next_long and delta > 1.75 * correct_long:
            timing_data = np.insert(timing_data, i + 1, timing_data[i] + correct_long)
            insertions += 1
            print("inserted missing timing pulse")

        if not next_long and delta > 1.75 * correct_short:
            timing_data = np.insert(timing_data, i + 1, timing_data[i] + correct_short)
            insertions += 1
            print("inserted missing timing pulse")

        next_long = not next_long
        i += 1

        if insertions > 100:
            print("aborting due to too many timing pulse insertions")
            break

    # plt.plot(timing_data)
    # plt.show()

    return timing_data


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
    raw = np.transpose(np.loadtxt(filename, delimiter=','))
    return dataset(short=raw[1], long=raw[0], time=raw[2], channel=raw[3])


show_n_gamma = False


def convert_mesy_file(in_file):
    in_handle = fileinput.input(in_file)

    longs, shorts, times, channels = [], [], [], []

    _ = next(in_handle)  # discard header line

    timeconst = 1 / 6.25e-8

    for line in in_handle:
        cols = line.split(",")
        time = int(float(cols[0]) * timeconst)
        for channel in range(16):
            long = cols[channel + 1]
            short = cols[channel + 17]
            if long != '':
                longs.append(float(long))
                shorts.append(float(short))
                times.append(time)
                channels.append(int(channel))

            channel += 1
    shorts = np.array(shorts)
    longs = np.array(longs)
    times = np.array(times)
    channels = np.array(channels)
    return dataset(shorts, longs, times, channels)


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


def PSD_Heatmap(data):
    #extract long_data from one channel (target channel = usecols), NaN values set to zero (NaN = other channel recorded event)
    long_data = np.genfromtxt(data, delimiter=',', skip_header=0, usecols=0, filling_values = 0)
    long_data = np.asarray(long_data).flatten()
    # print("Long Data is: " + str(len(long_data)))
    
    #extract short_datafrom one channel (target channel = usecols), NaN values set to zero (NaN = other channel recorded event)
    short_data = np.genfromtxt(data, delimiter=',', skip_header=0, usecols=1, filling_values = 0)
    short_data = np.asarray(short_data).flatten()
    # print("Short Data is: " + str(len(short_data)))

    #filter: long > short :long must be greater as short; makes no sense to have more in short than in long
    long_data_clean = []
    short_data_clean = []
    for i in range(len(long_data)):
        if (65535.0 > long_data[i] > short_data[i]):
            long_data_clean += [long_data[i]]
            short_data_clean += [short_data[i]]
    
    long_data_clean  = np.asarray(long_data_clean).flatten()
    short_data_clean = np.asarray(short_data_clean).flatten()
    # print(max(long_data_clean))

    return long_data_clean, short_data_clean


def analyze_run(data):
    data = data.subset(data.long > data.short)
    y = data.y()

    autorange = find_range(y, 100, 0.005)
    counts, bins = np.histogram(y, bins=100, range=autorange)
    bin_centers = bins[0:-1] + 0.5 * (bins[1] - bins[0])

    peaks = [-1, -1, -1]
    min_width = 2
    while len(peaks) > 2:
        peaks, _ = sig.find_peaks(counts, width=min_width, prominence=1.5, distance=min_width)
        min_width += 1

    peak_one = bin_centers[peaks[0]]
    peak_two = bin_centers[peaks[1]]

    sigma_guess_one = 1.7e-2
    sigma_guess_two = 1.7e-2

    amp_one = counts[peaks[0]]
    amp_two = counts[peaks[1]]

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
    cutoff_value = x_values[fitted_curve == np.min(fitted_curve[between_peaks])][0]

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


def load_file(filename, format_mesy=False, intended_lc=None, timing_chan=3, data_chan=2, n_gamma_co=0.0):
    data = None
    if format_mesy:
        data = convert_mesy_file(filename)
    else:
        data = load_dataset(filename)

    # fix timing overflows
    for i in range(1, len(data.time)):
        if data.time[i] < data.time[i - 1]:
            data.time[i:] += 2 ** 30

    # run n-gamma-discrimination
    sync_channel = timing_chan
    data_channel = data_chan
    timing_pulses = data.subset(data.channel == sync_channel).time
    if not intended_lc or len(timing_pulses) != 2 * intended_lc:
        timing_pulses = fix_timing_pulses(timing_pulses)
    print(f"number of timing pulses is: {len(timing_pulses)}")

    data = data.subset(data.channel == data_channel)
    data = data.subset(data.short < data.long) # measurement or transmission error check
    cutoff = n_gamma_co
    if cutoff == 0.:
        cutoff = analyze_run(data)
    neutron_hits = data.subset(data.y() > cutoff)
    gamma_hits = data.subset(data.y() <= cutoff)

    return neutron_hits, timing_pulses, gamma_hits


def hits_to_fluency(neutron_hits, timing_pulses, line_count, dt_timing=0.0):
    time_const = 6.25e-8  # assumes 12.5ns
    timing_pulse_const_offset = int(dt_timing / time_const)
    # convert hits to fluency
    times = neutron_hits.time

    # assosicate position values to data
    fluencies = []
    x_points = []
    y_points = []
    line_count = len(timing_pulses) // 2
    current_line = 0
    fwd = True
    prog_bar = progress_print.pbar(line_count, "binning lines ")
    for start, end in pairs(timing_pulses):
        start += timing_pulse_const_offset
        end -= timing_pulse_const_offset
        delta_t = end - start
        timeborders = np.linspace(start, end, line_count + 1)
        for i in range(0, line_count):
            mask = (times >= timeborders[i]) & (times <= timeborders[i + 1])
            fluencies.append(len(times[mask]) / delta_t)
            x_points.append(i if fwd else line_count - i - 1)
            y_points.append(current_line)
        prog_bar.next()
        current_line += 1
        fwd = not fwd

    fluencies = list(np.array(fluencies) / max(fluencies))

    # write result
    # csv_print(argv[2], x_points, y_points, fluencies)
    return (x_points, y_points, fluencies)
