import math


def square_scint(bin_count, size):
    high_bins = size * bin_count

    low_cut = int(math.ceil(high_bins / 2))
    high_cut = bin_count - low_cut

    xs = []
    ys = []
    amps = []
    for i in range(bin_count):
        for j in range(bin_count):
            xs.append(i)
            ys.append(j)
            amps.append(1.0 if (i < low_cut or i > high_cut) and (j < low_cut or j > high_cut) else 0.0)

    # csv_print(xs, ys, amps)
    return xs, ys, amps
