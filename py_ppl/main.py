#!python3
from sys import argv
import deconv
import data_loading
import square_scint


def main():
    data_file = argv[1]
    out_file = argv[2]
    mesy_format = bool(argv[3]) if len(argv) > 3 else False
    data = data_loading.load_file(data_file, mesy_format)
    scint = square_scint.square_scint(160, 30.)
    result = deconv.deconv(data, scint, 1000)
    print(result)
    print(out_file)
    # data_loading.csv_print(out_file, resul)


if __name__ == "__main__":
    main()
