import numpy as np


def square_scint(bin_count, size, area_size=30.):
    scint_ratio = size / area_size
    bins = bin_count * scint_ratio
    one_bins = int(np.floor(bins))
    # partial_bins = np.ceil(bins) - one_bins

    mat = np.zeros((bin_count, bin_count))

    for i in range(one_bins):
        mat[i][0:one_bins] += 1.0

    mat = np.roll(mat, - one_bins // 2, 0)
    mat = np.roll(mat, - one_bins // 2, 1)

    xs, ys, amps = [], [], []
    for i in range(bin_count):
        for j in range(bin_count):
            xs.append(i)
            ys.append(j)
            amps.append(mat[i][j])

    # csv_print(xs, ys, amps)
    return xs, ys, amps
