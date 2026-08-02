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

// Test: compile() produces the correct number of constraint rows when one
// path constraint is active. num_constraints = ni*ns (defects) + npc*nn (path).
// 4 intervals, 1 state, 0 alg => 4 defect rows. 1 path constraint, 5 nodes => 5 path rows.
TEST(HermiteSimpsonPath, CompileAddsPathConstraintRows) {
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
    ocp.num_path_constraints  = 1;
    ocp.path_constraint_lower = { 0.0 };
    ocp.path_constraint_upper = { goss::transcription::kInf };
    ocp.path_constraints = SingleStatePathConstraint{};

    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_path_rows");

    // 4 intervals, 1 state => 4 defect rows. 1 path constraint, 5 nodes => 5 path rows.
    const std::size_t expected_defects   = 4 * 1;
    const std::size_t expected_path_rows = 1 * 5;
    EXPECT_EQ(compiled.problem->num_constraints(), expected_defects + expected_path_rows);
}

// Test: path-constraint bounds in the NLPProblem match [0, +kInf] for >= constraint.
TEST(HermiteSimpsonPath, PathConstraintBoundsAreCorrect) {
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
    ocp.num_path_constraints  = 1;
    ocp.path_constraint_lower = { 0.0 };
    ocp.path_constraint_upper = { goss::transcription::kInf };
    ocp.path_constraints = SingleStatePathConstraint{};

    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_path_bounds");

    // Defect rows are indices 0..3 (4 defects). Path rows are indices 4..8 (5 nodes).
    // Each path row must have gl=0.0, gu=kInf.
    const std::size_t defect_count = 4;
    const auto& gl = compiled.problem->constraint_lower_bounds();
    const auto& gu = compiled.problem->constraint_upper_bounds();
    for (std::size_t path_row = 0; path_row < 5; ++path_row) {
        const std::size_t row_index = defect_count + path_row;
        EXPECT_DOUBLE_EQ(gl[row_index], 0.0)   << "path row " << path_row;
        EXPECT_DOUBLE_EQ(gu[row_index], goss::transcription::kInf) << "path row " << path_row;
    }
}

// Test: the packed functor evaluates path constraints at node 0 correctly under double.
// x(0) = 2.0 => g(x,u,t) = x[0] - 0.5 = 1.5 > 0 (constraint satisfied).
TEST(HermiteSimpsonPath, PackedFunctorEvaluatesPathConstraintAtNodeZero) {
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
    ocp.num_path_constraints  = 1;
    ocp.path_constraint_lower = { 0.0 };
    ocp.path_constraint_upper = { goss::transcription::kInf };
    ocp.path_constraints = SingleStatePathConstraint{};

    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_path_eval_node0");

    // Build a trial z: x_k = 2.0 for all nodes (ignoring dynamics).
    const std::size_t nv = compiled.problem->num_variables();
    std::vector<double> z_trial(nv, 2.0);

    const auto constraint_values = compiled.problem->eval_constraints(z_trial);
    // Path row for node 0, constraint 0 is at index defects + 0*npc + 0 = 4 + 0 = 4.
    // g = x[0] - 0.5 = 2.0 - 0.5 = 1.5
    EXPECT_NEAR(constraint_values[4], 1.5, 1e-12);
}

// Test: existing HermiteSimpson behaviour unchanged when num_path_constraints == 0.
// Uses the two-param OcpProblem form (NoPathConstraints sentinel deduced by default).
// Reuses file-scope SimpleDecayDynamics and ZeroCostFn to avoid local-struct
// template-deduction issues with the CppAD backend.
TEST(HermiteSimpsonPath, ZeroPathConstraintsLeavesRowCountUnchanged) {
    // Two-param form: AlgResFn and PathConstraintFn both default (no alg, no path).
    goss::transcription::OcpProblem<SimpleDecayDynamics, ZeroCostFn> ocp;
    ocp.num_states   = 1;
    ocp.num_controls = 0;
    ocp.dynamics     = SimpleDecayDynamics{};
    ocp.cost         = ZeroCostFn{};
    ocp.mesh         = goss::transcription::Mesh{0.0, 1.0, 4};
    ocp.state_lower  = { -goss::transcription::kInf };
    ocp.state_upper  = {  goss::transcription::kInf };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state       = { 1.0 };
    ocp.initial_state_fixed = { 1.0 };
    ocp.final_state         = { 0.0 };
    ocp.final_state_fixed   = { 0.0 };
    // num_path_constraints defaults to 0 — no path rows.

    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_path_zero");
    // 4 intervals * 1 state = 4 defect rows. 0 path rows.
    EXPECT_EQ(compiled.problem->num_constraints(), 4u);
}
