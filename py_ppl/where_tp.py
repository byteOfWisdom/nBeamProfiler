#!python3
import numpy as np
from matplotlib import pyplot as plt


raw_data = np.transpose(np.loadtxt("../lichtschranke_test.csv", delimiter=","))
tp = raw_data[2]
for i in range(1, len(tp)):
    if tp[i] < tp[i - 1]:
        tp[i:] += 2 ** 30

print(tp)

dt = tp[1:] - tp[:-1]
dt *= 1 / np.max(dt)
t_bar = 0.5 * (tp[1:] + tp[:-1])

plt.xlim(np.min(tp), np.max(tp))
plt.vlines(tp, 0, 1)
plt.scatter(t_bar, dt)
plt.show()
