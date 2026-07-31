// tests/ad/test_ad_backend_contract.cpp
#include <gtest/gtest.h>
#include <memory>
#include "goss/ad/ad_backend.hpp"
#include "goss/ad/errors.hpp"

namespace {
class StubBackend : public goss::ad::ADBackend {
    goss::ad::SparsityPattern empty_;
 public:
    std::size_t input_size() const override { return 2; }
    std::size_t output_size() const override { return 1; }
    std::vector<double> eval(const std::vector<double>&) const override { return {0.0}; }
    const goss::ad::SparsityPattern& jacobian_sparsity() const override { return empty_; }
    std::vector<double> eval_jacobian(const std::vector<double>&) const override { return {}; }
    const goss::ad::SparsityPattern& hessian_sparsity() const override { return empty_; }
    std::vector<double> eval_hessian(const std::vector<double>&, const std::vector<double>&) const override { return {}; }
};
}  // namespace

TEST(ADBackendContract, IsPolymorphicAndUsable) {
    std::unique_ptr<goss::ad::ADBackend> backend = std::make_unique<StubBackend>();
    EXPECT_EQ(backend->input_size(), 2u);
    EXPECT_EQ(backend->eval({1.0, 2.0}).size(), 1u);
}

TEST(ADBackendContract, ADErrorIsThrowable) {
    EXPECT_THROW(throw goss::ad::ADError("boom"), goss::ad::ADError);
}
