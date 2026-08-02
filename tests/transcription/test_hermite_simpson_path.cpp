// tests/transcription/test_hermite_simpson_path.cpp
// Unit tests for nonlinear path constraints at the HermiteSimpson transcription layer.
// Tests in this file use OcpProblem directly (no ExprModel) to isolate the
// transcription extension from the DSL layer.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"

namespace {

// Trivial path-constraint functor: g(x,u,t) = x[0] - 0.5  (enforces x >= 0.5)
// Returns a single-element vector so PathConstraintFn contract is satisfied.
struct SingleStatePathConstraint {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x,
                              const std::vector<T>& /*u*/,
                              T /*t*/) const {
        return { x[0] - T(0.5) };
    }
};

// Trivial path-constraint functor: always returns empty (no path constraints).
// Used to verify OcpProblem<Dyn,Cost> (two-param) still compiles.
struct NullConstraint {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& /*x*/,
                              const std::vector<T>& /*u*/,
                              T /*t*/) const {
        return {};
    }
};

struct SimpleDecayDynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x,
                              const std::vector<T>& /*u*/,
                              T /*t*/) const {
        return { -x[0] };
    }
};

struct ZeroCostFn {
    template <typename T>
    T operator()(const std::vector<T>& /*x*/,
                 const std::vector<T>& /*u*/,
                 T /*t*/) const {
        return T(0);
    }
};

}  // namespace

// Verify that OcpProblem<Dyn,Cost,AlgResFn,PathConstraintFn> can be constructed
// with one path constraint and that the path-constraint fields are accessible.
// AlgResFn is explicitly NoAlgebraicResiduals (4-param form) since the DAE
// follow-on already merged AlgResFn as the 3rd template param.
TEST(HermiteSimpsonPath, OcpProblemStoresPathConstraintFields) {
    goss::transcription::OcpProblem<
        SimpleDecayDynamics,
        ZeroCostFn,
        goss::transcription::NoAlgebraicResiduals,
        SingleStatePathConstraint> ocp;

    ocp.num_states   = 1;
    ocp.num_controls = 0;
    ocp.dynamics     = SimpleDecayDynamics{};
    ocp.cost         = ZeroCostFn{};
    ocp.mesh         = goss::transcription::Mesh{0.0, 1.0, 4};
    ocp.state_lower  = { -goss::transcription::kInf };
    ocp.state_upper  = {  goss::transcription::kInf };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state       = { 2.0 };
    ocp.initial_state_fixed = { 1.0 };
    ocp.final_state         = { 0.0 };
    ocp.final_state_fixed   = { 0.0 };

    // Algebraic fields keep their defaults (num_algebraic == 0).

    // Path-constraint fields (new in this task):
    ocp.num_path_constraints  = 1;
    ocp.path_constraint_lower = { 0.0 };   // g >= 0
    ocp.path_constraint_upper = { goss::transcription::kInf };
    ocp.path_constraints = SingleStatePathConstraint{};

    EXPECT_EQ(ocp.num_path_constraints, 1u);
    EXPECT_EQ(ocp.path_constraint_lower.size(), 1u);
    EXPECT_EQ(ocp.path_constraint_upper.size(), 1u);
}
