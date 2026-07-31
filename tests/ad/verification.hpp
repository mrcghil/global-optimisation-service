// tests/ad/verification.hpp
#pragma once
#include <complex>
#include <cstddef>
#include <vector>

namespace goss::ad::test {

template <typename F>
std::vector<std::vector<double>> finite_difference_jacobian(
    const F& f, const std::vector<double>& x, double step = 1e-6) {
    const std::size_t n = x.size();
    const std::size_t m = f(x).size();
    std::vector<std::vector<double>> jacobian(m, std::vector<double>(n, 0.0));
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<double> x_plus = x, x_minus = x;
        x_plus[j] += step;
        x_minus[j] -= step;
        auto y_plus = f(x_plus);
        auto y_minus = f(x_minus);
        for (std::size_t i = 0; i < m; ++i) {
            jacobian[i][j] = (y_plus[i] - y_minus[i]) / (2.0 * step);
        }
    }
    return jacobian;
}

template <typename F>
std::vector<std::vector<double>> complex_step_jacobian(
    const F& f, const std::vector<double>& x, double step = 1e-20) {
    using Complex = std::complex<double>;
    const std::size_t n = x.size();
    std::vector<Complex> x_complex(n);
    for (std::size_t j = 0; j < n; ++j) x_complex[j] = Complex(x[j], 0.0);
    const std::size_t m = f(x_complex).size();
    std::vector<std::vector<double>> jacobian(m, std::vector<double>(n, 0.0));
    for (std::size_t j = 0; j < n; ++j) {
        auto perturbed = x_complex;
        perturbed[j] = Complex(x[j], step);
        auto y = f(perturbed);
        for (std::size_t i = 0; i < m; ++i) {
            jacobian[i][j] = y[i].imag() / step;
        }
    }
    return jacobian;
}

}  // namespace goss::ad::test
