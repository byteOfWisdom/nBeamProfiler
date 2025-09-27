import numpy as np
from sys import argv
from matplotlib import pyplot as plt


data = np.transpose(np.loadtxt(argv[1], delimiter=","))
times = data[2]
time_offset = 2 ** 30
for i in range(1, len(times)):
    if times[i] < times[i - 1]:
        print(times[i] - times[i - 1])
        times[i:] += time_offset
delta_ts = times[1:] - times[:-1]

plt.plot(delta_ts)
plt.show()
