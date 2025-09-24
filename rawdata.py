from matplotlib import pyplot as plt
import numpy as np
from sys import argv

data = np.transpose(np.loadtxt(argv[1], delimiter=','))

x = data[0]
y = data[1]
flow = data[2]


lcount = 40
grid = [[0. for _ in range(lcount)] for _ in range(lcount)]

for i in range(len(x)):
    grid[int(x[i])][int(y[i])] = flow[i]

#plt.hist(x[y == y[int(len(y) / 2)]], bins=20)
plt.imshow(grid)
plt.show()
