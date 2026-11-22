from skimage import restoration
import numpy as np
from scipy.signal import convolve2d as conv2
from scipy.signal import fftconvolve, convolve2d
from matplotlib import pyplot as plt

def input_variance(data, psf, x_hat, mode="model"):
    """Poisson-Varianz der Eingangsdaten pro Pixel.

    mode="data":  Var = d  (Standard bei Poisson, aber bei 0 Counts exakt 0
                  -> unterschätzt die Unsicherheit in leeren Bins!)
    mode="model": Var = (A x_hat)_i, also die erwartete Zählrate laut Modell.
                  Leere Bins bekommen eine kleine, aber positive Varianz.
                  Das ist bei vielen Nullen die bessere Wahl.
    """
    if mode == "data":
        return data.astype(float)
    psf = psf / psf.sum()
    return np.clip(fftconvolve(x_hat, psf, mode="same"), 0, None)

# Richardson-Lucy Deconvolution algorithm from skimage package
def deconv_rl(measured, scint, iterations):
    return restoration.richardson_lucy(measured, scint, num_iter=iterations, clip=False)

# A.I. generated Richardson-Lucy Deconvolution with uncertainty propagation
def rl_with_uncertainty(image, psf, iterations=20, n_mc=300, read_noise_std=0.0, seed=42, variance="model"):
    
    rng = np.random.default_rng(seed)
    x_hat = restoration.richardson_lucy(image.astype(float), psf, iterations, clip=False)
    lam = input_variance(image, psf, x_hat, variance)
    samples = np.empty((n_mc,) + image.shape)
    for i in range(n_mc):
        samples[i] = restoration.richardson_lucy(rng.poisson(lam).astype(float), psf, iterations, clip=False)
    return samples.mean(axis=0), samples.std(axis=0, ddof=1)



# Re-convolve Deconvolution, compare with measured data and calculate sum of residuals
def residuals(measured, deconv_res, scint):
    reconv = np.array(conv2(deconv_res, scint, mode="same"))
    reconv_norm = np.sum(measured) / np.sum(reconv)
    reconv = reconv_norm * reconv
    return np.sum(np.abs(measured - reconv) ** 2)

# iterate with Richardson-Lucy algorithm until the sum of residuals between iteration steps deviate less then a cutoff value
def smart_deconv(measured, scint, iterations, close_to_zero=0.00001):
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
    # return deconv_rl(measured, scint, min_deviation), "smart deconv was used"
    return rl_with_uncertainty(measured, scint, iterations=min_deviation, n_mc=200), "smart deconv was used"
