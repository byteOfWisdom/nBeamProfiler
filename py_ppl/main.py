#!python3
from sys import argv

import deconv
import data_loading
import square_scint
import post_process
import visualization

import numpy as np
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt


def to_csv(mat):
    res = ""
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            res += f"{i}, {j}, {mat[i][j]}\n"
    return res


def get_assign(tag, default=None):
    for arg in argv:
        if arg.startswith(tag + "="):
            return "".join(arg.split("=")[1:])
    return default


def args_to_dict():
    res = {}
    res["in_file"] = get_assign("data")
    res["out_file"] = get_assign("out", "")
    res["lines"] = int(get_assign("lines", 160))
    res["timing_channel"] = int(get_assign("timing_chan", 3))
    res["data_channel"] = int(get_assign("data_chan", 2))
    res["iterations"] = int(get_assign("iter", 1000))
    res["format"] = get_assign("mesy_format", False)
    res["size"] = float(get_assign("size", 30.))
    res["scint_size"] = float(get_assign("scint_size", 2.54))
    res["preview"] = int(get_assign("preview", 0))
    res["n_gamma_cut"] = float(get_assign("n_gamma_cut", 0.))
    res["dt_timing"] = float(get_assign("dt_timing", 0.0))
    res["no_deconv"] = bool(get_assign("no_deconv", False))
    res["scint_amp_mod"] = float(get_assign("scint_amp_mod", 0.25))
    res["fit_beam_shape"] = bool(get_assign("fit_beam", False))
    return res


def missing_args(args):
    for a in args:
        if isinstance(get_assign(a), type(None)):
            return True
    return False


def matrix(arr):
    xdim = max(arr[0])
    ydim = max(arr[1])
    res = np.zeros((xdim + 1, ydim + 1))
    for x, y, v in zip(arr[0], arr[1], arr[2]):
        res[x][y] = v
    return res


def main():
    if missing_args(["data"]):
        print("missing relevant arguments")
        return
    args = args_to_dict()
    data_file = args['in_file']
    # out_file = args['out_file']
    mesy_format = args['format']
    print("---- loading input file ----")
    neutron_hits, timing_pulses, gamma_hits = data_loading.load_file(data_file,
                                                                     mesy_format,
                                                                     args['lines'],
                                                                     args['timing_channel'],
                                                                     args['data_channel'],
                                                                     args['n_gamma_cut'])

    data = data_loading.hits_to_fluency(neutron_hits, timing_pulses, args['lines'], args['dt_timing'])
    long_data , short_data = data_loading.PSD_Heatmap(data_file)
    neutron_hits, time_edges = np.histogram(neutron_hits.time, args['lines'] ** 2)

    if args['no_deconv']:
        plt.imshow(matrix(data))
        plt.show()
        return

    #DECONVOLUTION
    print("---- generating scintillator mask ----")
    scint = square_scint.square_scint(args["lines"], args["scint_size"], args['size'], args['scint_amp_mod'])
    print("---- running deconvolution ----")
    result, info = deconv.smart_deconv(matrix(data), scint, args["iterations"])
    print(info)

    #RE-CONVOLUTION
    reconvolved = np.array(conv2(result, scint, mode='same'))
    reconvolved_norm = np.array(reconvolved) / np.amax(reconvolved)
    print("---- Re-convolution successful ----")


    csv_data = to_csv(result)
    if args['out_file'] != "":
        with open(args['out_file'], "w") as handle:
            handle.write(csv_data)
    # print(csv_data)

    if args['fit_beam_shape']:
        print("---- attempting to fit a shape to the data ----")
        post_process.fit_beam(result)

    # previews:
    # - 1: Show 2D-Heatmaps
    # - 2: Show 3D-Contourplots
    # - 3: Show 2D-Heatmaps and 3D-Countourplots
    # - 4: Show Scan-Count Plot, PSD-Plot, 2D-Heatmaps and 3D-Countourplots
    # - 5: Export Scan-Count Plot and PSD-Plot
    # - 6: Export Beamscans
    if args['preview'] == 4:
        visualization.plot_c(time_edges, neutron_hits, timing_pulses, long_data, short_data, args)
    if args['preview'] == 1 or args['preview'] == 3 or args['preview'] == 4:
        visualization.plot_a(data, scint, reconvolved_norm, result, args)
    if args['preview'] == 2 or args['preview'] == 3 or args['preview'] == 4:
        diff1, diff2 = [], []
        visualization.plot_b(data, result, reconvolved_norm, diff1, diff2, args)
    if args['preview'] == 5:
            visualization.plot_d(time_edges, neutron_hits, timing_pulses, long_data, short_data, args, PGF=False)
    if args['preview'] == 6:
            visualization.plot_e(data, result, reconvolved_norm, args, PGF=False)

if __name__ == "__main__":
    main()
