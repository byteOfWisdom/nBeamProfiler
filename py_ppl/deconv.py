from skimage import restoration
import numpy as np
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt


def deconv_rl(measured, scint, iterations):
    return restoration.richardson_lucy(measured, scint, num_iter=iterations)


def residuals(measured, deconv_res, scint):
    reconv = np.array(conv2(deconv_res, scint, mode="same"))
    reconv_norm = np.sum(measured) / np.sum(reconv)
    reconv = reconv_norm * reconv
    return np.sum(np.abs(measured - reconv))


def smart_deconv(measured, scint, iterations, close_to_zero=0.0001):
    iter_residuals = [residuals(measured, deconv_rl(measured, scint, n), scint) for n in range(iterations)]
    delta = np.abs(np.gradient(iter_residuals))
    delta *= 1 / max(delta)
    min_deviation = iterations
    for i in range(len(delta)):
        if delta[i] < close_to_zero:
            min_deviation = i
            break

    print(f"sensible cutoff was found to be {min_deviation} iterations")
    # plt.plot(iter_residuals)
    # plt.plot(np.abs(delta))
    # plt.show()
    return deconv_rl(measured, scint, min_deviation), "smart deconv was used"
