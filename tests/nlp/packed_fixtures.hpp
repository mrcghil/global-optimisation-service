#pragma once
#include <cstddef>
#include <vector>
namespace goss::nlp::test {
// Output 0 = objective x0^2 + x1^2; output 1 = constraint x0 + x1 - 1.
struct QuadraticWithLinearConstraint {
    std::size_t input_size() const { return 2; }
    std::size_t output_size() const { return 2; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        return {x[0] * x[0] + x[1] * x[1], x[0] + x[1] - T(1)};
    }
};
// Output 0 = objective x0^2 + x1^2; output 1 = constraint x0^2 * x1 (nonlinear).
struct NonlinearConstraintProblem {
    std::size_t input_size() const { return 2; }
    std::size_t output_size() const { return 2; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        return {x[0] * x[0] + x[1] * x[1], x[0] * x[0] * x[1]};
    }
};
}  // namespace goss::nlp::test
