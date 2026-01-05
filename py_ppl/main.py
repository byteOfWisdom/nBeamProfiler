#!python3
from sys import argv
import deconv
import data_loading
import square_scint
import post_process
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


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
    data = data_loading.load_file(data_file,
                                  mesy_format,
                                  args['lines'],
                                  args['timing_channel'],
                                  args['data_channel'],
                                  args['n_gamma_cut'],
                                  args['dt_timing'])
    
    #extract long_data from one channel (target channel = usecols), NaN values set to zero (NaN = other channel recorded event)
    long_data = np.genfromtxt(data_file, delimiter=',', skip_header=0, usecols=0, filling_values = 0)
    long_data = np.asarray(long_data).flatten()
    # print("Long Data is: " + str(len(long_data)))
    
    #extract short_datafrom one channel (target channel = usecols), NaN values set to zero (NaN = other channel recorded event)
    short_data = np.genfromtxt(data_file, delimiter=',', skip_header=0, usecols=1, filling_values = 0)
    short_data = np.asarray(short_data).flatten()
    # print("Short Data is: " + str(len(short_data)))

    #filter: long > short :long must be greater as short; makes no sense to have more in short than in long
    long_data_clean = []
    short_data_clean = []
    for i in range(len(long_data)):
        if (long_data[i] > short_data[i]):
            long_data_clean += [long_data[i]]
            short_data_clean += [short_data[i]]
    
    long_data_clean  = np.asarray(long_data_clean).flatten()
    short_data_clean = np.asarray(short_data_clean).flatten()

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
    csv_data = to_csv(result)
    if args['out_file'] != "":
        with open(args['out_file'], "w") as handle:
            handle.write(csv_data)
    # print(csv_data)

    if args['fit_beam_shape']:
        print("---- attempting to fit a shape to the data ----")
        post_process.fit_beam(result)

    if args['preview'] == 1 or args['preview'] == 3:
        fig, ax = plt.subplots(2, 2)
        # plt.contour(result, levels=100)
        ax[0, 0].title.set_text("raw data")
        ax[0, 0].imshow(matrix(data))
        ax[0, 1].title.set_text("scint function")
        ax[0, 1].imshow(scint)
        ax[1, 0].title.set_text("PSD")
        ax[1, 0].hist2d(long_data_clean, ((long_data_clean - short_data_clean) / (long_data_clean)), bins=500, cmap='rainbow', norm=matplotlib.colors.LogNorm())
        # ax[1, 0].set_ylim(0.1,0.5)
        # ax[1, 0].imshow(np.zeros((2, 2)))
        ax[1, 1].title.set_text("unfolded data")
        ax[1, 1].imshow(result)
        fig.tight_layout()
        plt.show()
    if args['preview'] == 2 or args['preview'] == 3:
        ax = plt.figure().add_subplot(projection='3d')
        ax.view_init(elev=45, azim=-45, roll=0)
        # x, y = np.meshgrid(np.arange(np.max(data[0]) + 1), np.arange(np.max(data[1]) + 1))            #meshgrid with number of lines
        x, y = np.meshgrid(np.linspace(0,30,np.max(data[0]) + 1), np.linspace(0,30,np.max(data[1]) + 1))#meshgrid with lines to 30cm
        ax.contour(x, y, result, levels=100, axlim_clip=True)
        ax.contourf(x, y, result, zdir='x', offset=10, levels=300, cmap='rainbow', axlim_clip=True)
        # ax.contourf(x, y, result, zdir='y', offset=args['lines'], levels=10, cmap='rainbow', axlim_clip=True)  #projection with number of scan lines
        ax.contourf(x, y, result, zdir='y', offset=22, levels=300, cmap='rainbow', axlim_clip=True)               #projection with scaled to 30cm
        # ax.set_xlim(0,30)
        # ax.set_ylim(0,30)
        ax.set_zlim(0,1.1)
        ax.set_xlim(10,22)
        ax.set_ylim(10,22)
        ax.set_xlabel("x / cm")
        ax.set_ylabel("y / cm")
        ax.set_zlabel("normalised intensity")

        plt.show()
        # print(np.max(data[0]))
        # print(np.max(data[1]))
    # data_loading.csv_print(out_file, resul)


if __name__ == "__main__":
    main()
