#include <gtest/gtest.h>
#include "goss/nlp/errors.hpp"

TEST(NLPError, IsThrowable) {
    EXPECT_THROW(throw goss::nlp::NLPError("boom"), goss::nlp::NLPError);
}

#include <memory>
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/cppadcg_backend.hpp"
#include "nlp/packed_fixtures.hpp"

namespace {
std::unique_ptr<goss::ad::ADBackend> make_quad_backend(const std::string& name) {
    goss::nlp::test::QuadraticWithLinearConstraint f;
    return std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), name);
}
}  // namespace

TEST(NLPConstruction, ReportsDimensions) {
    goss::nlp::NLPProblem problem(make_quad_backend("nlp_dims"),
        {-10.0, -10.0}, {10.0, 10.0}, {0.0}, {0.0});
    EXPECT_EQ(problem.num_variables(), 2u);
    EXPECT_EQ(problem.num_constraints(), 1u);
    EXPECT_EQ(problem.variable_lower_bounds().size(), 2u);
    EXPECT_EQ(problem.constraint_upper_bounds().size(), 1u);
}

TEST(NLPConstruction, RejectsMismatchedVariableBounds) {
    EXPECT_THROW(
        goss::nlp::NLPProblem(make_quad_backend("nlp_badvar"),
            {-10.0}, {10.0, 10.0}, {0.0}, {0.0}),
        goss::nlp::NLPError);
}

TEST(NLPConstruction, RejectsWrongConstraintCount) {
    // backend has 1 constraint (output_size 2); supplying 2 constraint bounds is wrong
    EXPECT_THROW(
        goss::nlp::NLPProblem(make_quad_backend("nlp_badcon"),
            {-10.0, -10.0}, {10.0, 10.0}, {0.0, 0.0}, {0.0, 0.0}),
        goss::nlp::NLPError);
}

TEST(NLPConstruction, RejectsInvertedBounds) {
    EXPECT_THROW(
        goss::nlp::NLPProblem(make_quad_backend("nlp_inv"),
            {10.0, -10.0}, {-10.0, 10.0}, {0.0}, {0.0}),  // var0 lower>upper
        goss::nlp::NLPError);
}
