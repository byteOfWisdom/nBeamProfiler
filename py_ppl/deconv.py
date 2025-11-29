from juliacall import Main as jl
import numpy as np
from skimage import restoration

jl.seval("using DeconvOptim")



def deconv(measured, scint, iterations):
    # res, meta = jl.deconvolution(measured, scint, regularizer=jl.TV(), iterations=iterations)
    res, meta = jl.deconvolution(measured, scint, iterations=iterations, loss=jl.Gauss())
    return np.array(res), meta


def deconv_rl(measured, scint, iterations):
    return restoration.richardson_lucy(measured, scint, num_iter=iterations), "richardson lucy was used"
