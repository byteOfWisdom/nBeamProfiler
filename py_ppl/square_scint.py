import numpy as np


def only_scint(bin_count, size, area_size=30.):
    scint_ratio = size / area_size
    bins = bin_count * scint_ratio
    one_bins = int(np.floor(bins))
    partial_bins = 0.5 * (bins - one_bins)

    mat = np.zeros((one_bins + 2, one_bins + 2))
    for i in range(0, one_bins + 2):
        mat[i][0:one_bins + 2] += partial_bins

    for i in range(1, one_bins + 1):
        mat[i][1:one_bins + 1] += 1.0 - partial_bins

    x, y = np.meshgrid(np.linspace(-1, 1, one_bins + 2), np.linspace(-1, 1, one_bins + 2))
    efficiency_mask = np.exp(- (x**2 + y**2) / 0.5)
    efficiency_mask *= 1 / np.max(efficiency_mask)
    efficiency_mask *= 0.5

    # mat = mat * efficiency_mask
    mat -= efficiency_mask
    mat *= 1 / np.max(mat)
    return mat


def square_scint(bin_count, size, area_size=30.):
    mat = np.zeros((bin_count, bin_count))

    os = only_scint(bin_count, size, area_size)
    return os
    x, y = np.shape(os)

    for i in range(x):
        for j in range(y):
            mat[i][j] = os[i][j]

    xs, ys, amps = [], [], []
    for i in range(bin_count):
        for j in range(bin_count):
            xs.append(i)
            ys.append(j)
            amps.append(mat[i][j])

    # csv_print(xs, ys, amps)
    return xs, ys, amps
