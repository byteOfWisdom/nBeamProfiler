#!python3
from sys import argv
import deconv
import data_loading
import square_scint


def to_csv(mat):
    res = ""
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            res += f"{i}, {j}, {mat[i][j]}\n"


def get_assign(tag, default=None):
    for arg in argv:
        if arg.startswith(tag + "="):
            return "".join(arg.split("=")[1:])
    return default


def args_to_dict():
    res = {}
    res["in_file"] = get_assign("data")
    res["out_file"] = get_assign("out")
    res["lines"] = int(get_assign("lines", 40))
    res["timing_channel"] = int(get_assign("timing_chan", 3))
    res["data_channel"] = int(get_assign("data_chan", 1))
    res["iterations"] = int(get_assign("iter", 1000))
    res["format"] = get_assign("mesy_format", False)
    res["size"] = float(get_assign("size", 30.))
    return res


def missing_args(args):
    for a in args:
        if isinstance(get_assign(a), type(None)):
            return True
    return False


def main():
    if missing_args(["data", "out"]):
        print("missing relevant arguments")
        return
    args = args_to_dict()
    data_file = args['in_file']
    out_file = args['out_file']
    mesy_format = args['format']
    data = data_loading.load_file(data_file, mesy_format)
    scint = square_scint.square_scint(args["lines"], args["size"])
    result = deconv.deconv(data, scint, args["iterations"])
    # print(result)
    print(out_file)
    csv_data = to_csv(result)
    print(csv_data)
    # data_loading.csv_print(out_file, resul)


if __name__ == "__main__":
    main()
