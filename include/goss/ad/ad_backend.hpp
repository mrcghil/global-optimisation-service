// include/goss/ad/ad_backend.hpp
#pragma once
#include <vector>
#include "goss/ad/types.hpp"
namespace goss::ad {
class ADBackend {
 public:
    virtual ~ADBackend() = default;
    virtual std::size_t input_size() const = 0;
    virtual std::size_t output_size() const = 0;
    virtual std::vector<double> eval(const std::vector<double>& x) const = 0;
    virtual const SparsityPattern& jacobian_sparsity() const = 0;
    virtual std::vector<double> eval_jacobian(const std::vector<double>& x) const = 0;
    virtual const SparsityPattern& hessian_sparsity() const = 0;
    virtual std::vector<double> eval_hessian(const std::vector<double>& x,
                                             const std::vector<double>& weights) const = 0;
};
}  // namespace goss::ad
