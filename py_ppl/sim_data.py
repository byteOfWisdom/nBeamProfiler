#!python3
import numpy as np
from matplotlib import pyplot as plt
import square_scint
from sys import argv
import progress_print


def beam_func(x, y):
    return np.exp(-(((x**2 + y**2) / 6) ** 4))


def absmin(a, b):
    return a if np.abs(a) < np.abs(b) else b


def time_point_gen(dt):
    t = 0
    speed = 5
    vert_step = 30.0 / 160
    x, y, t = 0, 0, 0
    to_x, to_y = 0.0, 0
    current_line = 0
    time_pulse = False
    fwd = True
    horizontal = True
    prog_bar = progress_print.pbar(160, "simulating scan ")
    while 1:
        mx, my = to_x - x, to_y - y
        t += dt
        if mx:
            x += absmin(mx, speed * dt * (mx / np.abs(mx)))
        elif my:
            y += absmin(my, speed * dt * (my / np.abs(my)))
        else:
            time_pulse = True
            if fwd and horizontal:
                fwd = not fwd
                horizontal = False
                to_x = 30.0
            elif not fwd and horizontal:
                fwd = not fwd
                horizontal = False
                to_x = 0.0
            elif not horizontal:
                horizontal = True
                current_line += 1
                prog_bar.next()
                to_y += vert_step

        yield t, x, y, time_pulse
        time_pulse = False

        if current_line >= 160:
            return None


sigma_gamma_sq = 0.0007910551062305672
sigma_gamma = np.sqrt(sigma_gamma_sq)
sigma_neutron_sq = 0.0006662172224860486
sigma_neutron = np.sqrt(sigma_gamma_sq)
mu_gamma = 0.32564104036173314
mu_neutron = 0.44286772376157973


def main():
    xs, ys = np.meshgrid(np.linspace(0, 30.0, 160), np.linspace(0, 30.0, 160))
    beam = beam_func(xs, ys)

    rng = np.random.default_rng()

    raw_scint = square_scint.square_scint(160, 2.54)
    scint = np.zeros((160, 160))
    i, j = np.meshgrid(
        np.arange(np.shape(raw_scint)[1]), np.arange(np.shape(raw_scint)[0])
    )
    scint[i, j] += raw_scint[i, j]

    scint = np.roll(scint, 80, 0)
    scint = np.roll(scint, 80, 1)
    time_const = 1 / 6.25e-8

    ax = plt.figure().add_subplot(projection="3d")
    ax.contour(xs, ys, scint, levels=100, cmap=plt.cm.viridis)
    ax.contour(xs, ys, beam, levels=100, cmap=plt.cm.inferno)
    plt.show()

    dt = 0.05

    file = open(argv[1], "w")

    total_hits = 0

    for t, x, y, tp in time_point_gen(dt):
        if tp:
            file.write(f"{5}, {1}, {int(t * time_const)}, {3}\n")
        hit_count = int(1e3 * np.sum(beam_func(xs - x, ys - y) * scint))
        # print(hit_count)
        total_hits += hit_count
        neutron_ys = iter(rng.normal(mu_neutron, sigma_neutron, hit_count))
        gamma_ys = iter(rng.normal(mu_gamma, sigma_gamma, hit_count))
        for tc in time_const * np.linspace(t, t + dt, num=hit_count, endpoint=False):
            long = 1000
            file.write(
                f"{long}, {int(long * (1 - next(neutron_ys)))}, {int(tc)}, {2}\n"
            )
            file.write(f"{long}, {int(long * (1 - next(gamma_ys)))}, {int(tc)}, {2}\n")

    file.close()
    print(f"total neutron hit count is {total_hits}")


if __name__ == "__main__":
    main()
