import cffi
import tempfile


ffi_builder = cffi.FFI()
ffi_builder.cdef("double pulse_func(double t, double a, double b, double tau_0, double tau_1, double t0, double sigma_0, double sigma_1);")
ffi_builder.cdef("void v_pulse_func(int n, double* t, double* out, double a, double b, double tau_0, double tau_1, double t0, double sigma_0, double sigma_1);")

c_code = None
with open("./pulse_func.c", "r") as handle:
    c_code = handle.read()

ffi_builder.set_source("_pulse_func", source=c_code)

if __name__ == "__main__":
    # ffi_builder.compile(tmpdir=tempfile.gettempdir(), target="./_pulse_func.*", verbose=True)
    ffi_builder.compile(verbose=True)
