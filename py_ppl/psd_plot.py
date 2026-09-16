from matplotlib import pyplot as plt
import data_loading
from sys import argv
import sciebo_fetch
import numpy as np
import scipy


# a = 2.85880836e+08 
# b = 3.23529442e-09
# c = -9.10068348e-01


def energy_to_ch(x):
    a = 3.20180374e+04  
    b = 5.02109330e-01 
    c = -1.00000000e+01
    return a * np.log(b * x + 1) + c* x


def ch_to_energy(x):
    a = 3.20180374e+04  
    b = 5.02109330e-01 
    c = -1.00000000e+01
    return np.real((a * b * scipy.special.lambertw(c * np.exp(c / (a * b) + x / a) / (a * b)) - c) / (b * c))


print(energy_to_ch(4.2))

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

    fig, ax = plt.subplots()

    ax.hist2d(dataset.long, dataset.y(), bins=(bin_count, bin_count), range=((0, max(dataset.long)), (0, max(dataset.y()[dataset.y() < 1]))))
    ax.set_xlabel("long / channel")
    ax.set_ylabel("$Q$")
    energy_axis = ax.secondary_xaxis("top", functions=(ch_to_energy, energy_to_ch))
    energy_axis.set_xlabel("energy / MeV")

    plt.show()
