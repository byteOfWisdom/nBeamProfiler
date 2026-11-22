from skimage import restoration
import numpy as np
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve


# Richardson-Lucy Deconvolution algorithm from skimage package
def deconv_rl(measured, scint, iterations):
    return restoration.richardson_lucy(measured, scint, num_iter=iterations, clip=False)

# A.I. generated Richardson-Lucy Deconvolution with uncertainty propagation
# - deconvolute your image 'iterations'-times obtain your result 
# - create 'n_mc' synthetic datasets by sampling from Poisson distribution
# - deconvolute synthetic samples using identical parameters
# - calculate pixel-wise standard deviation
def rl_with_uncertainty(image, psf, iterations=20, n_mc=100, norm_factor=1.0, read_noise_std=0.0):
    """
    Computes Richardson-Lucy deconvolution and pixel-wise standard deviation.
    Assumes 'image' is in photon/electron counts (ADU * gain).
    """
    # 1. Primary deconvolution
    deconv_main = restoration.richardson_lucy(image, psf, num_iter=iterations, clip=False)

    # 2. Re-convolve with PSF to get a denoised estimate of expected counts
    # Ensure non-negative rate for Poisson sampling by np.maximum()
    # Re-normalize by *norm_factor
    expected_counts = np.maximum(fftconvolve(deconv_main, psf, mode="same"), 0.0) *norm_factor

    # 3. Bootstrap in physical count space
    rng = np.random.default_rng(seed=42) # set random seed for reproducable results

    stack = np.empty((n_mc, *image.shape), dtype=np.float32)
    for i in range(n_mc):
        # sim_counts = rng.poisson(rate).astype(np.float64)
        sim_counts = rng.poisson(expected_counts).astype(np.float64) # Sample synthetic realizations from expected_counts instead of raw counts
        stack[i] = restoration.richardson_lucy(sim_counts, psf, num_iter=iterations, clip=False)

    uncertainty_norm = np.std(stack, axis=0, ddof=1) / norm_factor

    
    # print(str(iterations) + " were performed to calculate the Unfolding uncertainty")
    return deconv_main, uncertainty_norm

def propagate_rl_uncertainty(image: np.ndarray, psf: np.ndarray, num_iter: int = 20, n_samples: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
  """
    Propagates Poisson uncertainties (sigma = sqrt(counts)) through 2D
    Richardson-Lucy deconvolution using Monte Carlo resampling.

    Parameters
    ----------
    image : 2D ndarray
        2D array of binned counts where variance == counts (sigma = sqrt(counts)).
    psf : 2D ndarray
        2D Point Spread Function (will be normalized to sum to 1).
    num_iter : int
        Number of Richardson-Lucy iterations.
    n_samples : int
        Number of Monte Carlo realizations (200–500 recommended for smooth error maps).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    deconvolved : 2D ndarray
        Deconvolved 2D image.
    uncertainty : 2D ndarray
        Pixel-wise 1-sigma uncertainty map in the same units as `image`.
    """
  rng = np.random.default_rng(seed)

  # Ensure non-negativity and normalized PSF
  image_clean = np.maximum(image, 0.0)
  psf_norm = psf / np.sum(psf)

  # 1. Deconvolve the actual observed image
  # clip=False prevents scikit-image from clipping values outside [-1, 1]
  deconvolved = restoration.richardson_lucy(
      image_clean, psf_norm, num_iter=num_iter, clip=False
  )

  # 2. Generate synthetic realizations with sigma = sqrt(image) and deconvolve each
  realizations = np.empty((n_samples, *image.shape), dtype=np.float32)

  for i in range(n_samples):
    # Poisson draw has mean = image_clean and std = sqrt(image_clean)
    synthetic_image = rng.poisson(image_clean).astype(np.float64)
    # synthetic_image = rng.normal(image_clean).astype(np.float64)
    realizations[i] = restoration.richardson_lucy(
        synthetic_image, psf_norm, num_iter=num_iter, clip=False
    )

  # 3. Pixel-wise standard deviation across all deconvolved realizations
  uncertainty = np.std(realizations, axis=0, ddof=1)

  return deconvolved, uncertainty

# Re-convolve Deconvolution, compare with measured data and calculate sum of residuals
def residuals(measured, deconv_res, scint):
    reconv = np.array(conv2(deconv_res, scint, mode="same"))
    reconv_norm = np.sum(measured) / np.sum(reconv)
    reconv = reconv_norm * reconv
    return np.sum(np.abs(measured - reconv) ** 2)

# iterate with Richardson-Lucy algorithm until the sum of residuals between iteration steps deviate less then a cutoff value
def smart_deconv(measured, scint, iterations, normConstant=1.0, close_to_zero=0.00001):
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
    # return rl_with_uncertainty(measured, scint, iterations=min_deviation, n_mc=250, norm_factor=normConstant), "smart deconv was used"
    return propagate_rl_uncertainty(image=measured, psf=scint, num_iter=min_deviation, n_samples=200, seed=42), "smart deconv was used"
