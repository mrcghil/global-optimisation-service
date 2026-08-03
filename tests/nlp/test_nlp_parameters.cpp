#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/nlp/nlp_problem.hpp"

namespace {
struct ParamObjOnly {  // output 0 = objective, no constraints
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& z, const std::vector<T>& p) const {
        return { p[0] * z[0] * z[0] };
    }
};
}  // namespace

TEST(NlpParameters, ForwardsCountAndInjectionToBackend) {
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(
        ParamObjOnly{}, /*input_size=*/1, /*parameter_size=*/1,
        std::vector<double>{1.0}, "nlp_param_obj");
    goss::nlp::NLPProblem problem(std::move(backend),
        /*var_lb=*/{-1e19}, /*var_ub=*/{1e19},
        /*con_lb=*/{}, /*con_ub=*/{});

    ASSERT_EQ(problem.num_parameters(), 1u);
    problem.set_parameters({2.0});
    EXPECT_NEAR(problem.eval_objective({3.0}), 2.0 * 9.0, 1e-12);   // 18
    problem.set_parameters({10.0});
    EXPECT_NEAR(problem.eval_objective({3.0}), 10.0 * 9.0, 1e-12);  // 90
}
