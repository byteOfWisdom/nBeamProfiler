import scipy
import numpy as np


def supergaussian(r, amp, mu_x, mu_y, sigma_x, sigma_y, power):
    x, y = r
    exp_term = - ((x - mu_x) ** 2 / (sigma_x ** 2)) - ((y - mu_y) ** 2 / (sigma_y ** 2))
    return amp * np.exp(exp_term ** power)
    

def fit_beam(data):
    x = np.linspace(0, 30, len(data))
    y = np.linspace(0, 30, len(data))
    xg, yg = np.meshgrid(x, y)
    r_data = np.vstack((xg.ravel(), yg.ravel()))
    z_data = data.ravel()

    p0 = [np.max(data), 15., 15., 1, 1, 8]

    params, cov = scipy.optimize.curve_fit(supergaussian, r_data, z_data, maxfev=int(1e5))
    print(params)
    return None
