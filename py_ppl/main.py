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
    plt.plot(0.5 * (time_edges[1:] + time_edges[:-1]), neutron_hits)
    plt.show()

    if args['no_deconv']:
        plt.imshow(matrix(data))
        plt.show()
        return

    print("---- generating scintillator mask ----")
    scint = square_scint.square_scint(args["lines"], args["scint_size"], args['size'], args['scint_amp_mod'])
    print("---- running deconvolution ----")
    result, info = deconv.deconv_rl(matrix(data), scint, args["iterations"])
    # print(result)
    print(info)
    # print(data)
    # print(out_file)

    reconvolved = np.array(conv2(result, scint, mode='same'))
    # print(np.amax(reconvolved))
    reconvolved_norm = np.array(reconvolved) / np.amax(reconvolved)
    # print(np.amax(reconvolved_norm))
    print("Re-convolution successful")

    diff1 = []
    diff2 = []
    result_old =[]
    result_next =[]
    reconvolved_old = []
    reconvolved_next = []
    for i in range(args["iterations"]+1):
        result_next, info2 = deconv.deconv_rl(matrix(data), scint, i)
        reconvolved_next = np.array(conv2(result_next, scint, mode='same'))
        reconvolved_next = np.array(reconvolved_next) / np.amax(reconvolved_next)
        if i > 0:
            # print( np.sum( (result_next - result_old)**2 ) )
            diff1 = np.append(diff1, np.sqrt(np.sum( (result_next - result_old)**2) ))
            diff2 = np.append(diff2, np.sqrt(np.sum( (reconvolved_old - matrix(data))**2) ))
        result_old = result_next
        reconvolved_old = reconvolved_next
        # print(diff1)
        # print(diff2)
        # print(info2 + " for " + str(i) + " times")
    min_index = np.argmin(diff2)
    print("Minimum reached after " + str(min_index+1) + " iterations with " + str(diff2[min_index]))


    csv_data = to_csv(result)
    if args['out_file'] != "":
        with open(args['out_file'], "w") as handle:
            handle.write(csv_data)
    # print(csv_data)

    if args['fit_beam_shape']:
        print("---- attempting to fit a shape to the data ----")
        post_process.fit_beam(result)

    if args['preview'] == 1 or args['preview'] == 3:
        visualization.plot_a(data, scint, long_data, short_data, result, args)
    if args['preview'] == 2 or args['preview'] == 3:
        visualization.plot_b(data, result, reconvolved_norm, diff1, diff2, args)


if __name__ == "__main__":
    main()
