#!python3
from sys import argv
import deconv
import data_loading
import square_scint
import numpy as np
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
    res["lines"] = int(get_assign("lines", 40))
    res["timing_channel"] = int(get_assign("timing_chan", 3))
    res["data_channel"] = int(get_assign("data_chan", 1))
    res["iterations"] = int(get_assign("iter", 1000))
    res["format"] = get_assign("mesy_format", False)
    res["size"] = float(get_assign("size", 30.))
    res["scint_size"] = float(get_assign("scint_size", 2.54))
    res["preview"] = int(get_assign("preview", 0))
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
    data = data_loading.load_file(
                                  data_file,
                                  mesy_format,
                                  args['lines'],
                                  args['timing_channel'],
                                  args['data_channel'])
    print("---- generating scintillator mask ----")
    scint = square_scint.square_scint(args["lines"], args["scint_size"], args['size'])
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

    if args['preview'] == 1 or args['preview'] == 3:
        fig, ax = plt.subplots(2, 2)
        # plt.contour(result, levels=100)
        ax[0, 0].imshow(matrix(data))
        ax[0, 1].imshow(scint)
        ax[1, 0].imshow(np.zeros((2, 2)))
        ax[1, 1].imshow(result)
        plt.show()
    if args['preview'] == 2 or args['preview'] == 3:
        ax = plt.figure().add_subplot(projection='3d')
        x, y = np.meshgrid(np.arange(np.max(data[0]) + 1), np.arange(np.max(data[1]) + 1))
        ax.contour(x, y, result, levels=100)
        plt.show()
    # data_loading.csv_print(out_file, resul)


if __name__ == "__main__":
    main()
