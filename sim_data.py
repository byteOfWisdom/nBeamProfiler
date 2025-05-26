# %%
from typing import Tuple
from typing_extensions import Callable
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from numba import njit

# %%
vx = 4.
vy = 1.
ax = 100.
ay = 100.
nrows = 40

@njit
def velo(t: float) -> Tuple[float, float]:
    total_time = (ax * nrows / vx) + (ay / vy)
    while t > total_time:
        t -= total_time

    tpr = total_time / nrows
    xtime = ax / vx
    fwd = True
    while t > tpr:
        t -= tpr
        fwd = not fwd
    if t > xtime:
        return 0, vy
    if fwd:
        return vx, 0
    else:
        return - vx, 0

@njit
def velo_x( t: float) -> float:
    x, _ = velo(t)
    return x

@njit
def velo_y( t: float) -> float:
    _, y = velo(t)
    return y

@njit
def zig_zag_path(t: float):
    x_pos = 0.
    y_pos = 0.
    dt = 0.01
    count = 0.
    while count < t:
        dvx, dvy = velo(count)
        count += dt
        x_pos += dvx * dt
        y_pos += dvy * dt

    return x_pos, y_pos


@njit
def area_at(pos: Tuple[float, float], shape: Callable[[float, float], float]) ->  Callable[[float, float], float]:
    return lambda x, y: shape(pos[0] - x, pos[1] - y)

scint_size = 40.
@njit
def square(x: float, y: float):
    #return np.exp(- (x**2 + y**2) / (50000.)) ** 1000.
    return 1. if x ** 2 < scint_size ** 2 and y ** 2 < scint_size ** 2 else 0.


# area should be a function that gives a aperature at time t

@njit(parallel=True)
def estimate_hits(
    shape: Callable[[float, float], float],
    path: Callable[[float], Tuple[float, float]],
    beam_shape: Callable[[float, float], float],
    t_max: float, dt: float, points: np.array, dr: float, rng: bool
) -> Tuple[np.array, np.array]:
    cnt = int(t_max / dt)
    res = []
    times = np.random.random_sample(cnt) * t_max if rng else np.linspace(0, t_max, cnt)
    for t in times:
        px, py = path(t)
        scint = lambda x, y: shape(x - px, y - py)
        value = sum([scint(a[0], a[1]) * beam_shape(a[0], a[1]) * dr for a in points])
        res.append(value)
    return np.array(res), times


@njit
def sample_beam(x: float, y: float) -> float:
    amp = 10.
    sigma_sq_x = 70.
    sigma_sq_y = 1000.
    return amp * np.exp(- (((x - 50.) ** 2) / (2 * sigma_sq_x)) - (((y - 50.) ** 2) / (2 * sigma_sq_y)))


# %%

path = zig_zag_path
beam_shape = sample_beam

data = np.transpose(np.array(list(map(path, np.linspace(0, 1100, 1000)))))
#print(data)
y, x = np.meshgrid(np.linspace(0, 100, 1000), np.linspace(0, 100, 1000))
"""
plt.pcolormesh(x, y, beam_shape(x, y))
plt.scatter(data[0], data[1], marker='.')
axes=plt.gca()
axes.set_aspect(1)
plt.show()
"""

# %%
x, y = np.meshgrid(np.linspace(0, 100, 1000), np.linspace(0, 100, 1000))
dr = 100 / 1000
hit_map, times = estimate_hits(square, zig_zag_path, sample_beam, 1100, 0.1, np.transpose(np.vstack([x.ravel(), y.ravel()])), dr, False)

data = np.transpose(np.array(list(map(path, times))))
psf_values = [square(x[0] - 50., x[1] - 50.) for x in np.transpose(data)]

np.savetxt("sample_gauss_even_intervall.csv", np.transpose([times, hit_map, data[0], data[1], psf_values]))

hit_map, times = estimate_hits(square, zig_zag_path, sample_beam, 1100, 0.1, np.transpose(np.vstack([x.ravel(), y.ravel()])), dr, True)

data = np.transpose(np.array(list(map(path, times))))
np.savetxt("sample_gauss_random_intervall.csv", np.transpose([times, hit_map, data[0], data[1], psf_values]))
exit()
#"""
plt.plot(times, hit_map)
plt.xlabel("time")
plt.ylabel("count rate")
plt.grid()
plt.show()
#"""
# %%

# correlate timestamps to positions
# may be more complicated with real data
positions = np.transpose(np.array([zig_zag_path(t) for t in times]))
# so now we have c (x, y)

#"""
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.bar3d(positions[0], positions[1], hit_map, dx=1.5, dy=1.5, dz =3.5)
plt.show()
#"""

# %%
@njit
def range_2d(xcnt, ycnt):
    for x in range(xcnt):
        for y in range(ycnt):
            yield (x, y)


@njit
def smaple_grid(positions: np.array, values: np.array, nrows: int):
    xbins, ybins = nrows, nrows
    xmax, ymax = 100., 100.
    # assume data ist in form of rows
    # slice into rows:
    rows_pos = np.split(positions, nrows)
    rows_values = np.split(values, nrows)

    hist = [[[0.] for _ in range(ybins)] for __ in range(xbins)]
    res = [[0. for _ in range(ybins)] for __ in range(xbins)]
    # now group each row into bins by x position
    xstep = xmax / (xbins)
    for y in range(ybins):
        pos, v = rows_pos[y], rows_values[y]
        for n in range(len(pos)):
            xbin = int(pos[n][0] // xstep) - 1
            hist[xbin][y].append(v[n])

    for x,y in range_2d(xbins, ybins):
        res[x][y] = sum(hist[x][y]) / len(hist[x][y])

    return res


grid = smaple_grid(np.array([zig_zag_path(t) for t in times]), hit_map, nrows)

plt.imshow(np.transpose(grid))
axes=plt.gca()
axes.set_aspect(1)
plt.show()
# %%
@njit
def populate_shape_grid(xbins, ybins, xmax, ymax):
    res = [[0. for _ in range(ybins)] for __ in range(xbins)]
    xstep = xmax / xbins
    ystep = ymax / ybins
    for x, y in range_2d(xbins, ybins):
        res[x][y] = square((x * xstep) - 50., (y * ystep) - 50.)
    return res


def reconstruct_beam_shape(grid, nrows):
    x, y = np.meshgrid(np.linspace(0, 100., nrows), np.linspace(0, 100., nrows))
    ft_grid = np.fft.fft2(grid)
    ft_shape = np.fft.fft2(populate_shape_grid(nrows, nrows, 100., 100.))

    #valid = np.abs(ft_shape) > 0.0000001
    frac = ft_grid / ft_shape
    #frac[valid] = ft_grid[valid] / ft_shape[valid]

    final = np.fft.ifft2(frac)
    print(ft_grid)
    return final

plt.imshow(np.transpose(np.abs(reconstruct_beam_shape(grid, nrows))))
plt.show()


# %%
#
# alternate crossection reconstruction:
from skimage import restoration

def deconvolve(star, psf):
    return restoration.richardson_lucy(np.array(star), np.array(psf), num_iter=350)

plt.imshow(np.abs(np.transpose(deconvolve(grid, populate_shape_grid(nrows, nrows, 100., 100.)))))
plt.show()
