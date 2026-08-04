import cffi
import tempfile

# # @np.vectorize
# @numba.njit
# def mod_gaussian(t, t_r, sigma, tau):
#     dt = t - t_r
#     # mask = np.heaviside(2 * (sigma ** 2) + dt * tau, 0)
#     mask = 2 * (sigma ** 2) + dt * tau > 0
#     if mask:
#         return np.exp(- dt ** 2 / (2 * (sigma ** 2) + tau * dt))
#     else:
#         return 0

# @np.vectorize
# @numba.njit
# def f_egh(t, H, tau, t_r, sigma):
#     # dt = t - t_r
#     # mask = np.heaviside(2 * (sigma ** 2) + dt * tau, 0)
#     f_0 = mod_gaussian(t, t_r, sigma, tau)
#     # f_0[mask == 0] = 0
#     return H * f_0


# @np.vectorize
# # @numba.njit
# def pulse_function(t, a, b, tau_0, tau_1, t0,  sigma_0, sigma_1):
#     return f_egh(t, a, tau_0, t0, sigma_0) + f_egh(t, b, tau_1, t0, sigma_1)


ffi_builder = cffi.FFI()
ffi_builder.cdef("double pulse_func(double t, double a, double b, double tau_0, double tau_1, double t0, double sigma_0, double sigma_1);")

c_code = None
with open("./pulse_func.c", "r") as handle:
    c_code = handle.read()

ffi_builder.set_source("_pulse_func", source=c_code)

if __name__ == "__main__":
    # ffi_builder.compile(tmpdir=tempfile.gettempdir(), target="./_pulse_func.*", verbose=True)
    ffi_builder.compile(verbose=True)
