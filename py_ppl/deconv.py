from juliacall import Main as jl
import numpy as np

jl.seval("using DeconvOptim")


def deconv(measured, scint, iterations):
    res, meta = jl.deconvolution(measured, scint, regularizer=jl.Tikhonov(), iterations=iterations)
    return np.array(res), meta
