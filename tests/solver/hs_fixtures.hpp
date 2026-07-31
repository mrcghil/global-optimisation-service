// tests/solver/hs_fixtures.hpp
// Hand-ported Hock-Schittkowski test problems as packed functors for IPOPT
// integration tests.
//
// Packing convention: output[0] = objective, output[1..m] = constraints.
// Constraint bounds encode the constraint sense (see NLPProblem):
//   - equality h(x) = 0   -> bound [0, 0]
//   - inequality g(x) >= c -> bound [c, kInf]
//   - inequality g(x) <= c -> bound [-kInf, c]
#pragma once
#include <cstddef>
#include <vector>

namespace goss::solver::hs {

/// Sentinel value for one-sided constraint / variable bounds (open side).
constexpr double kInf = 2e19;

// ---------------------------------------------------------------------------
// HS71 — canonical IPOPT example
// Objective : min x1*x4*(x1+x2+x3) + x3
// Inequality: x1*x2*x3*x4 >= 25  (output 1, bound [25, kInf])
// Equality  : x1^2+x2^2+x3^2+x4^2 = 40  (output 2, bound [40, 40])
// Variable bounds: 1 <= xi <= 5
// x0 = (1, 5, 5, 1);  x* ≈ (1.0, 4.743, 3.821, 1.379);  f* ≈ 17.0140173
// ---------------------------------------------------------------------------
struct HS71 {
    std::size_t input_size() const { return 4; }
    std::size_t output_size() const { return 3; }

    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        T obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        T g   = x[0] * x[1] * x[2] * x[3];                       // >= 25
        T h   = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];  // == 40
        return {obj, g, h};
    }
};

// ---------------------------------------------------------------------------
// HS28 — equality-only QP
// Objective : min (x1+x2)^2 + (x2+x3)^2
// Equality  : x1 + 2*x2 + 3*x3 = 1  (output 1, bound [0, 0])
// Variable bounds: free ([-kInf, kInf])
// x0 = (-4, 1, 1);  x* = (0.5, -0.5, 0.5);  f* = 0
// ---------------------------------------------------------------------------
struct HS28 {
    std::size_t input_size() const { return 3; }
    std::size_t output_size() const { return 2; }

    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        T obj = (x[0] + x[1]) * (x[0] + x[1]) + (x[1] + x[2]) * (x[1] + x[2]);
        // Constraint packed as residual h = x1 + 2*x2 + 3*x3 - 1; bound [0,0]
        T h   = x[0] + T(2) * x[1] + T(3) * x[2] - T(1);
        return {obj, h};
    }
};

// ---------------------------------------------------------------------------
// HS35 — inequality-only QP
// Objective : min 9 - 8*x1 - 6*x2 - 4*x3 + 2*x1^2 + 2*x2^2 + x3^2
//                 + 2*x1*x2 + 2*x1*x3
// Inequality: x1 + x2 + 2*x3 <= 3  (output 1, bound [-kInf, 3])
// Variable bounds: xi >= 0
// x0 = (0.5, 0.5, 0.5);  f* = 1/9 ≈ 0.1111
// ---------------------------------------------------------------------------
struct HS35 {
    std::size_t input_size() const { return 3; }
    std::size_t output_size() const { return 2; }

    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        T obj = T(9) - T(8)*x[0] - T(6)*x[1] - T(4)*x[2]
              + T(2)*x[0]*x[0] + T(2)*x[1]*x[1] + x[2]*x[2]
              + T(2)*x[0]*x[1] + T(2)*x[0]*x[2];
        T g   = x[0] + x[1] + T(2)*x[2];  // <= 3
        return {obj, g};
    }
};

}  // namespace goss::solver::hs
