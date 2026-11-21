from skimage import restoration
import numpy as np
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt


# Richardson-Lucy Deconvolution algorithm from skimage package
def deconv_rl(measured, scint, iterations):
    return restoration.richardson_lucy(measured, scint, num_iter=iterations)

# A.I. generated Richardson-Lucy Deconvolution with uncertainty propagation
def rl_with_uncertainty(image, psf, iterations=20, n_mc=300, read_noise_std=0.0):
    """
    Computes Richardson-Lucy deconvolution and pixel-wise standard deviation.
    Assumes 'image' is in photon/electron counts (ADU * gain).
    """
    # 1. Primary deconvolution
    deconv_main = restoration.richardson_lucy(image, psf, num_iter=iterations, clip=True)
    
    # 2. Online variance accumulation (Welford's algorithm)
    rng = np.random.default_rng(seed=42)
    mean = np.zeros_like(deconv_main, dtype=np.float64)
    m2 = np.zeros_like(deconv_main, dtype=np.float64)
    
    # Ensure non-negative rate for Poisson sampling
    rate = np.clip(image, 0, None)
    
    for i in range(1, n_mc + 1):
        # Simulate noisy observation
        sim_image = rng.poisson(rate).astype(np.float64)
        if read_noise_std > 0:
            sim_image += rng.normal(0, read_noise_std, size=image.shape)
            
        # Deconvolve using identical parameters
        sim_deconv = restoration.richardson_lucy(sim_image, psf, num_iter=iterations, clip=True)
        
        # Update running mean and sum of squared differences
        delta = sim_deconv - mean
        mean += delta / i
        delta2 = sim_deconv - mean
        m2 += delta * delta2

    pixel_std = np.sqrt(m2 / (n_mc - 1))
    print(str(iterations) + " were performed to calculate the Unfolding uncertainty")
    return deconv_main, pixel_std

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
    return rl_with_uncertainty(measured, scint, iterations=min_deviation, n_mc=250), "smart deconv was used"
