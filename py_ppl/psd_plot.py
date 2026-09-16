from matplotlib import pyplot as plt
import data_loading
from sys import argv
import sciebo_fetch
import numpy as np
import scipy
from types import NoneType


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


if __name__ == "__main__":
    if len(argv) < 3:
        print('python psd_plot.py ["local" | URL of sciebo] [filename | "ls"] [OPTONAL ARGS ...]')
        print(' [filename] may be replaced by "ls" to list and interactively pick a file from a sciebo archive')
        print('\toptional args:')
        print(' --bins [number] : sets both x and y direction bin counts to that number')
        print(' --cut [number] : display the neutron / gamma discrimination line at given value')
        exit(0)

    source = argv[1]
    name = argv[2]
    bin_count = 100
    if "--bins" in argv:
        bin_count = int(argv[1 + argv.index("--bins")])

    n_gamma_cut = None
    if "--cut" in argv:
        n_gamma_cut = float(argv[1 + argv.index("--cut")])

    load_handle = None
    if source != "local":
        data_store = sciebo_fetch.fetch(source, True)
        if name == "ls":
            files = data_store.ls()
            for i in range(len(files)):
                print(f"[{i}] {files[i]}")
            name = files[int(input("select file by number: "))]
        load_handle = data_store.np_loadable(name)
    else:
        load_handle = name

    long, short, t, c, _ = np.genfromtxt(load_handle, delimiter=",", unpack=True)
    dataset = data_loading.dataset(short, long, t, c)
    dataset = dataset.subset(dataset.long > dataset.short)

    fig, ax = plt.subplots()

    particle_hist = ax.hist2d(dataset.long, dataset.y(), bins=(bin_count, bin_count), range=((0, max(dataset.long)), (0, max(dataset.y()[dataset.y() < 1]))), cmap="plasma")
    ax.set_xlabel("long / channel")
    ax.set_ylabel("$Q$")
    energy_axis = ax.secondary_xaxis("top", functions=(ch_to_energy, energy_to_ch))
    energy_axis.set_xlabel("energy / MeV")
    fig.colorbar(particle_hist[3])

    if not isinstance(n_gamma_cut, NoneType):
        ax.axhline(n_gamma_cut, linestyle="dashed", color="white")
        ax.text(max(dataset.long) - 10000, n_gamma_cut + 0.02, "neutrons", color="white")
        ax.text(max(dataset.long) - 10000, n_gamma_cut - 0.03, "gammas", color="white")

    plt.show()
