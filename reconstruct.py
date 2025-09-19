import numpy as np
from matplotlib import pytplot as plt
form sys import argv


def main():
    data = np.transpose(np.loadtxt(argv[1]))
    times = data[0]
    fluency = data[1]
    xlist = data[2]
    ylist = data[3]



if __name__ == "__main__":
    main()
