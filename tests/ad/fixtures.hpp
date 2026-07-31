// tests/ad/fixtures.hpp
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

namespace goss::ad::test {

struct Quadratic {
    std::size_t n;
    explicit Quadratic(std::size_t num_inputs) : n(num_inputs) {}
    std::size_t input_size() const { return n; }
    std::size_t output_size() const { return 1; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        T accumulator = T(0);
        for (const auto& value : x) accumulator += value * value;
        return {T(0.5) * accumulator};
    }
};

struct Rosenbrock {
    std::size_t input_size() const { return 2; }
    std::size_t output_size() const { return 1; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        T a = T(1) - x[0];
        T b = x[1] - x[0] * x[0];
        return {a * a + T(100) * b * b};
    }
};

struct Banded {
    std::size_t n;
    explicit Banded(std::size_t size) : n(size) {}
    std::size_t input_size() const { return n; }
    std::size_t output_size() const { return n; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        std::vector<T> y(n);
        for (std::size_t i = 0; i < n; ++i) {
            T next = (i + 1 < n) ? x[i + 1] : T(0);
            y[i] = x[i] * x[i] + next;  // depends on x[i], x[i+1]
        }
        return y;
    }
};

struct Trig {
    std::size_t input_size() const { return 2; }
    std::size_t output_size() const { return 2; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        using std::sin; using std::cos;
        return {sin(x[0]) * x[1], cos(x[1])};
    }
};

}  // namespace goss::ad::test
