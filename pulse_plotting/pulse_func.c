#include <math.h>

// # # @np.vectorize
// # @numba.njit
// # def mod_gaussian(t, t_r, sigma, tau):
// #     dt = t - t_r
// #     # mask = np.heaviside(2 * (sigma ** 2) + dt * tau, 0)
// #     mask = 2 * (sigma ** 2) + dt * tau > 0
// #     if mask:
// #         return np.exp(- dt ** 2 / (2 * (sigma ** 2) + tau * dt))
// #     else:
// #         return 0

// # @np.vectorize
// # @numba.njit
// # def f_egh(t, H, tau, t_r, sigma):
// #     # dt = t - t_r
// #     # mask = np.heaviside(2 * (sigma ** 2) + dt * tau, 0)
// #     f_0 = mod_gaussian(t, t_r, sigma, tau)
// #     # f_0[mask == 0] = 0
// #     return H * f_0


// # @np.vectorize
// # # @numba.njit
// # def pulse_function(t, a, b, tau_0, tau_1, t0,  sigma_0, sigma_1):
// #     return f_egh(t, a, tau_0, t0, sigma_0) + f_egh(t, b, tau_1, t0, sigma_1)

double pulse_func(double t, double a, double b, double tau_0, double tau_1, double t0, double sigma_0, double sigma_1);
double mod_gaussian(double t, double t_r, double sigma, double tau);
double f_egh(double t, double H, double tau, double t_r, double sigma);
void v_pulse_func(int n, double* t, double* out, double a, double b, double tau_0, double tau_1, double t0, double sigma_0, double sigma_1);

double mod_gaussian(double t, double t_r, double sigma, double tau) {
      double dt = t - t_r;
      double mask = 2 * sigma * sigma + dt * tau;
      if (mask <= 0) return 0.;
      return exp(- dt * dt / mask);
}

double f_egh(double t, double H, double tau, double t_r, double sigma) {
      return H * mod_gaussian(t, t_r, sigma, tau);
}


double pulse_func(double t, double a, double b, double tau_0, double tau_1, double t0, double sigma_0, double sigma_1) {
      return f_egh(t, a, tau_0, t0, sigma_0) + f_egh(t, b, tau_1, t0, sigma_1);
  }


void v_pulse_func(int n, double* t, double* out, double a, double b, double tau_0, double tau_1, double t0, double sigma_0, double sigma_1) {
      for (int i = 0; i < n; ++i) out[i] = pulse_func(t[i], a, b, tau_0, tau_1, t0, sigma_0, sigma_1);
}
