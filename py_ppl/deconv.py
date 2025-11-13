from juliacall import Main as jl

jl.seval("using DeconvOptim")


def deconv(measured, scint, iterations):
    res, meta = jl.deconvolution(measured, scint, regularizer=jl.Tikhonov(), iterations=iterations)
    return res, meta
