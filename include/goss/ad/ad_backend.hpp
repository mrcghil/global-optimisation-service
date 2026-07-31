// include/goss/ad/ad_backend.hpp
#pragma once
#include <vector>
#include "goss/ad/types.hpp"
namespace goss::ad {
class ADBackend {
 protected:
    ADBackend() = default;

 public:
    virtual ~ADBackend() = default;
    ADBackend(const ADBackend&) = delete;
    ADBackend& operator=(const ADBackend&) = delete;
    ADBackend(ADBackend&&) = delete;
    ADBackend& operator=(ADBackend&&) = delete;

    virtual std::size_t input_size() const = 0;
    virtual std::size_t output_size() const = 0;
    virtual std::vector<double> eval(const std::vector<double>& x) const = 0;

    /// Returns the (row, col) index pairs of the Jacobian in the SAME ORDER
    /// that eval_jacobian() returns its values: result[k] is the sparsity
    /// entry for eval_jacobian()[k].
    virtual const SparsityPattern& jacobian_sparsity() const = 0;

    /// Evaluates the Jacobian at x.  result[k] is the partial derivative
    /// identified by jacobian_sparsity()[k].
    virtual std::vector<double> eval_jacobian(const std::vector<double>& x) const = 0;

    /// Returns the (row, col) index pairs for the LOWER TRIANGLE of the
    /// Hessian (col <= row), in the SAME ORDER that eval_hessian() returns
    /// its values: result[k] is the sparsity entry for eval_hessian()[k].
    virtual const SparsityPattern& hessian_sparsity() const = 0;

    /// Evaluates the weighted-sum Hessian H = sum_i weights[i] * d²f_i/dx²
    /// at x.  Only the lower triangle (col <= row) is returned.  result[k]
    /// is the Hessian value at the index pair hessian_sparsity()[k].
    virtual std::vector<double> eval_hessian(const std::vector<double>& x,
                                             const std::vector<double>& weights) const = 0;
};
}  // namespace goss::ad
