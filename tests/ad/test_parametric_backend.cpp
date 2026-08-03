// tests/ad/test_parametric_backend.cpp
//
// Tests for the parametric CppADCGBackend constructor (pinned-variable design).
// The backend records f over a combined domain z = [x, p], JIT-compiles ONCE,
// and exposes only the x-part to callers via input_size() / eval() /
// eval_jacobian() / eval_hessian().  Parameters are injected via
// set_parameters() and spliced into the combined vector at every evaluation.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/ad/errors.hpp"

namespace {

// f(z; p) = p0 * z0^2 + z1
// grad_z f = [2 * p0 * z0,  1]
// Hess_z f = [[2 * p0,  0],
//             [0,       0]]  (lower-triangle: (0,0) = 2*p0)
struct ParamQuadratic {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& z,
                              const std::vector<T>& p) const {
        return {p[0] * z[0] * z[0] + z[1]};
    }
};

}  // namespace

// -------------------------------------------------------------------------
// EvalTracksInjectedParameter
//   Verifies that changing the injected parameter changes the evaluated value
//   WITHOUT re-recording or re-compiling the model.
// -------------------------------------------------------------------------
TEST(ParametricBackend, EvalTracksInjectedParameter) {
    goss::ad::CppADCGBackend backend(
        ParamQuadratic{}, /*input_size=*/2, /*parameter_size=*/1,
        /*parameter_defaults=*/{1.0}, "param_quad_eval");

    ASSERT_EQ(backend.num_parameters(), 1u);
    // input_size() must report only the decision-variable count (2), not 3.
    ASSERT_EQ(backend.input_size(), 2u);

    // With p0 = 3 and x = [2, 7]:  f = 3 * 4 + 7 = 19
    backend.set_parameters({3.0});
    EXPECT_NEAR(backend.eval({2.0, 7.0})[0], 19.0, 1e-12);

    // With p0 = 5 and x = [2, 7]:  f = 5 * 4 + 7 = 27
    backend.set_parameters({5.0});
    EXPECT_NEAR(backend.eval({2.0, 7.0})[0], 27.0, 1e-12);
}

// -------------------------------------------------------------------------
// JacobianTracksParameter
//   Verifies that the Jacobian w.r.t. x only (not p) is returned, and that
//   its values reflect the currently injected parameter.
//   With p0 = 4, x = [2, 7]:
//     d/dz0 = 2 * 4 * 2 = 16
//     d/dz1 = 1
// -------------------------------------------------------------------------
TEST(ParametricBackend, JacobianTracksParameter) {
    goss::ad::CppADCGBackend backend(
        ParamQuadratic{}, 2, 1, {1.0}, "param_quad_jac");

    backend.set_parameters({4.0});

    const auto& pattern = backend.jacobian_sparsity();
    const auto jac = backend.eval_jacobian({2.0, 7.0});

    // Pattern and values must be aligned and cover only x-columns (col < 2).
    ASSERT_EQ(pattern.size(), jac.size());

    // The Jacobian has exactly two non-zeros: (row=0,col=0) and (row=0,col=1).
    // Both columns must be strictly < input_size (= 2) — param column dropped.
    for (const auto& [row, col] : pattern) {
        EXPECT_LT(col, backend.input_size())
            << "Jacobian sparsity contains a param-column entry: col=" << col;
    }

    bool found_col0 = false;
    bool found_col1 = false;
    for (std::size_t k = 0; k < pattern.size(); ++k) {
        if (pattern[k].second == 0) {
            EXPECT_NEAR(jac[k], 16.0, 1e-9)
                << "d/dz0 should be 2*p0*z0 = 2*4*2 = 16";
            found_col0 = true;
        }
        if (pattern[k].second == 1) {
            EXPECT_NEAR(jac[k], 1.0, 1e-9)
                << "d/dz1 should be 1";
            found_col1 = true;
        }
    }
    EXPECT_TRUE(found_col0) << "Jacobian is missing col=0 entry";
    EXPECT_TRUE(found_col1) << "Jacobian is missing col=1 entry";
}

// -------------------------------------------------------------------------
// SetParametersRejectsWrongSize
//   Verifies that set_parameters() throws ADError when given the wrong
//   number of values.
// -------------------------------------------------------------------------
TEST(ParametricBackend, SetParametersRejectsWrongSize) {
    goss::ad::CppADCGBackend backend(
        ParamQuadratic{}, 2, 1, {1.0}, "param_quad_badsz");

    // Providing two values for a backend with parameter_size=1 must throw.
    EXPECT_THROW(backend.set_parameters({1.0, 2.0}), goss::ad::ADError);
}
