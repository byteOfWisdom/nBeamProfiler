from matplotlib import pyplot as plt
import data_loading
from sys import argv
import sciebo_fetch
import numpy as np


if __name__ == "__main__":
    if len(argv) < 3:
        print('["local" | URL of sciebo] [filename] [OPTONAL ARGS ...]')
        print('\toptional args:')
        print(' --bins [number] : sets both x and y direction bin counts to that number')
        print(' --Ecal [number] [number] : slope and offset for a linear relation from channels to energies. slope as eV / Channel')

    source = argv[1]
    name = argv[2]
    bin_count = 100
    if "--bins" in argv:
        bin_count = int(argv[1 + argv.index("--bins")])

    has_ecal = False
    ecal = None
    if "--Ecal" in argv:
        has_ecal = True
        slope = float(argv[1 + argv.index("--Ecal")])
        offset = float(argv[2 + argv.index("--Ecal")])
        def linear_func(ch): return ch * slope + offset
        ecal = linear_func

    load_handle = None
    if source != "local":
        data_store = sciebo_fetch.fetch(source, True)
        load_handle = data_store.np_loadable(name)
    else:
        load_handle = name

    long, short, t, c, _ = np.genfromtxt(load_handle, delimiter=",", unpack=True)
    dataset = data_loading.dataset(short, long, t, c)
    dataset = dataset.subset(dataset.long > dataset.short)

    plt.hist2d(dataset.long, dataset.y(), bins=(bin_count, bin_count), range=((0, max(dataset.long)), (0, max(dataset.y()[dataset.y() < 1]))))

    if has_ecal:
        pass
    plt.show()
