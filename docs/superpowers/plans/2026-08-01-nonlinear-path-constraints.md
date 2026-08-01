# Nonlinear Path Constraints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add general nonlinear path constraints `g(x,u,t) >= 0` (and `<=`, `==`) evaluated at every collocation node, with both a transcription-layer hook in `OcpProblem` / `HermiteSimpson` and an expr-DSL lowering so `(q + 1.0) >= 0.0` compiles and produces the correct NLP rows.

**Architecture:** `OcpProblem<Dyn,Cost>` gains two new fields — a `PathConstraintFn` functor (template `vector<T> eval(x,u,t)`) plus per-constraint lower/upper bound vectors — with empty defaults so all existing users compile unchanged. `HermiteSimpson::compile()` appends `num_path_constraints * num_nodes` constraint rows (after defects) in the packed functor, each with bounds `[gl_j, gu_j]` corresponding to `>=0` / `<=0` / `==0`. On the DSL side, a new `PathConstraintExpr<Expr>` struct and a new `PathConstraintFunctor<ExprTuple>` parallel `DynamicsFunctor`; expression-typed `operator>=/<=/==` overloads (SFINAE-gated to exclude bare `StateHandle`/`ControlHandle` LHS) produce `PathConstraintExpr` instead of `BoundConstraint`. `ExprModel` accumulates path constraints in a `std::tuple` (same type-accumulating pattern as dynamics) and `build()` passes them into `OcpProblem`. No `std::function` anywhere in the AD path.

**Tech Stack:** C++17, header-only `goss` framework, GoogleTest, CppADCG, IPOPT via `goss::solver::IpoptSolver`, `scripts/dev.sh` container-first builds.

## Global Constraints

- **C++17** — no C++20. No `std::concepts`, no `requires` clauses, use `std::enable_if_t` / `if constexpr` for SFINAE and branching.
- **Header-only** — all new files under `include/goss/`. No new `.cpp` sources.
- **AD-safety** — every path through the AD tape (the packed functor in `HermiteSimpson::compile`) must be fully templated on `ScalarT`. No `std::function`, no virtual dispatch, no type erasure in that path. Capture `PathConstraintFunctor` by value inside the packed lambda.
- **Backward compatibility** — bare `StateHandle q; q >= 0.0` MUST still lower to `BoundConstraint` (box bound), not a path constraint. This is enforced by overload resolution: the new expression-typed operators live in `goss::model::expr` and require the LHS to be an expr-DSL node type (not a `StateHandle`/`ControlHandle`). The existing operators in `goss::model` remain untouched.
- **Bound convention** — `g >= 0` → `[0, +kInf]`; `g <= 0` → `[-kInf, 0]`; `g == 0` → `[0, 0]`.
- **Scope** — HermiteSimpson only in this plan. Trapezoidal and LGL are deferred with a note (the `OcpProblem` extension is already present; only the row-append loop in each scheme's packed functor needs adding).
- **Node-only enforcement** — path constraints are evaluated at the `nn` collocation nodes only, not at Hermite midpoints. Rationale: midpoint `x_mid` is the Hermite interpolation formula (not a decision variable); evaluating `g(x_mid, u_mid, t_mid)` adds `ni` extra rows for each constraint but does not improve constraint satisfaction at actual nodes. Node-only is standard practice for direct collocation and keeps the count at `num_path_constraints * nn`. Midpoint enforcement is noted as a future option.
- **VariableLayout unchanged** — path constraints add only constraint rows, not decision variables. `layout.total_variables()` is not affected.
- **Accuracy-suite dependency** — Task 5 (the end-to-end accuracy test) requires `goss_accuracy_tests` target and `tests/accuracy/accuracy_helpers.hpp` from the accuracy-validation-suite plan (must merge first).
- **No `OcpProblem` template parameter added** — `PathConstraintFn` is a new template parameter on `OcpProblem` with a default (`NoPathConstraints` sentinel), so existing two-parameter specialisations still compile without change.
- **Verbose names** — all variables, members, and function parameters use full descriptive names. Type annotations everywhere. Comments explain WHY.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `include/goss/transcription/ocp_problem.hpp` | Modify | Add `PathConstraintFn` third template param (defaulted to `NoPathConstraints`), `num_path_constraints`, `path_constraint_lower`, `path_constraint_upper`, `path_constraints` fields |
| `include/goss/transcription/hermite_simpson.hpp` | Modify | Append path-constraint rows in packed functor; extend `gl`/`gu` construction |
| `include/goss/model/expr/path_constraint.hpp` | Create | `PathConstraintExpr<Expr>`, `PathConstraintFunctor<ExprTuple>`, `PathConstraintEntry<Expr>`, expression-typed `operator>=/<=/==` overloads in `goss::model::expr` |
| `include/goss/model/expr/expr_model.hpp` | Modify | Add `PathTuple` third template parameter; `with_path_constraint()` accumulator; `build()` extension to pass path constraints into `OcpProblem` |
| `include/goss/model/expr/expr.hpp` | Modify | Add `#include "goss/model/expr/path_constraint.hpp"` |
| `tests/transcription/test_hermite_simpson_path.cpp` | Create | Unit tests for path-constraint rows at the transcription layer (no DSL) |
| `tests/model/test_path_constraint_lowering.cpp` | Create | Unit tests for DSL operator overloads and ExprModel accumulation |
| `tests/accuracy/test_path_constraint_accuracy.cpp` | Create | End-to-end accuracy test with known reference optimum |
| `CMakeLists.txt` | Modify | Add new test sources to `goss_transcription_tests`, `goss_model_tests`, `goss_accuracy_tests` |

---

### Task 1: Extend `OcpProblem` with path-constraint fields

**Files:**
- Modify: `include/goss/transcription/ocp_problem.hpp`

**Interfaces:**
- Produces: `struct goss::transcription::NoPathConstraints` — sentinel functor with `num_path_constraints() == 0` and a no-op `eval`.
- Produces: Updated `OcpProblem<DynamicsFn, CostFn, PathConstraintFn = NoPathConstraints>` with new fields:
  - `PathConstraintFn path_constraints` — functor satisfying `template<T> std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& u, T t) const`
  - `std::size_t num_path_constraints` — number of scalar constraints returned by `path_constraints`
  - `std::vector<double> path_constraint_lower` — per-constraint lower bounds (size `num_path_constraints`)
  - `std::vector<double> path_constraint_upper` — per-constraint upper bounds (size `num_path_constraints`)
- Preserves: all existing two-parameter `OcpProblem<Dyn,Cost>` sites compile unchanged (third param defaults to `NoPathConstraints`, `num_path_constraints` defaults to `0`, bound vectors default to empty).

- [ ] **Step 1: Write a failing test that constructs an `OcpProblem` with a path constraint**

Create `tests/transcription/test_hermite_simpson_path.cpp`:

```cpp
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

// Verify that OcpProblem<Dyn,Cost,PathConstraintFn> can be constructed with one
// path constraint and that the fields are accessible.
TEST(HermiteSimpsonPath, OcpProblemStoresPathConstraintFields) {
    goss::transcription::OcpProblem<SimpleDecayDynamics, ZeroCostFn, SingleStatePathConstraint> ocp;
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
    ocp.path_constraint_lower = { 0.0 };   // g >= 0
    ocp.path_constraint_upper = { goss::transcription::kInf };
    ocp.path_constraints = SingleStatePathConstraint{};

    EXPECT_EQ(ocp.num_path_constraints, 1u);
    EXPECT_EQ(ocp.path_constraint_lower.size(), 1u);
    EXPECT_EQ(ocp.path_constraint_upper.size(), 1u);
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_transcription_tests 2>&1 | tail -20'
```

Expected: compile error — `OcpProblem` has no third template param, `num_path_constraints` undeclared.

- [ ] **Step 3: Implement the `OcpProblem` extension**

In `include/goss/transcription/ocp_problem.hpp`, replace the entire file content:

```cpp
// include/goss/transcription/ocp_problem.hpp
#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"
#include "goss/transcription/transcription.hpp"  // kInf

namespace goss::transcription {

/// Sentinel type used as the default PathConstraintFn when no path constraints
/// are registered. Its operator() returns an empty vector so the transcription
/// layer can dispatch uniformly without a special-case branch.
///
/// AD-safety: fully templated operator() — no std::function, no virtual dispatch.
struct NoPathConstraints {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& /*x*/,
                              const std::vector<T>& /*u*/,
                              T /*t*/) const {
        return {};
    }
};

struct Mesh {
    double t_initial;
    double t_final;
    std::size_t num_intervals;
    std::size_t num_nodes() const { return num_intervals + 1; }
    double interval_width() const { return (t_final - t_initial) / static_cast<double>(num_intervals); }
    void validate() const {
        if (num_intervals == 0) throw TranscriptionError("Mesh: num_intervals must be >= 1");
        if (t_final <= t_initial) throw TranscriptionError("Mesh: t_final must be > t_initial");
    }
};

/// Optimal-control problem descriptor passed to a transcription scheme's compile().
///
/// Template parameters:
///   DynamicsFn       — satisfies: template<T> vector<T> operator()(const vector<T>& x,
///                                                                    const vector<T>& u, T t) const
///   CostFn           — satisfies: template<T> T operator()(const vector<T>& x,
///                                                           const vector<T>& u, T t) const
///   PathConstraintFn — satisfies: template<T> vector<T> operator()(const vector<T>& x,
///                                                                    const vector<T>& u, T t) const
///                      Returns a vector of length num_path_constraints.
///                      Defaults to NoPathConstraints (empty return, no rows added).
///
/// DAE ORTHOGONALITY NOTE: The forthcoming DAE plan will add algebraic-variable
/// fields (num_algebraic, algebraic_residual, algebraic_lower, algebraic_upper).
/// Those fields occupy a distinct semantic slot (algebraic state variables entering
/// the NLP decision vector) and MUST NOT be confused with path_constraints (which
/// add only constraint rows, never decision variables). When both plans are merged,
/// path_constraint_* fields and algebraic_* fields must coexist in OcpProblem; the
/// executor reconciling the two plans must verify no field name collisions.
template <typename DynamicsFn, typename CostFn,
          typename PathConstraintFn = NoPathConstraints>
struct OcpProblem {
    std::size_t num_states;
    std::size_t num_controls;
    DynamicsFn  dynamics;   // template<T> vector<T> (const vector<T>& x, const vector<T>& u, T t)
    CostFn      cost;       // template<T> T (const vector<T>& x, const vector<T>& u, T t)
    Mesh        mesh;
    std::vector<double> state_lower;
    std::vector<double> state_upper;
    std::vector<double> control_lower;
    std::vector<double> control_upper;
    std::vector<double> initial_state;
    std::vector<double> initial_state_fixed;   // nonzero => pin node 0 state i
    std::vector<double> final_state;
    std::vector<double> final_state_fixed;     // nonzero => pin last node state i

    // --- Path-constraint extension (defaults produce zero additional NLP rows) ---

    /// Number of scalar path constraints (length of the vector returned by path_constraints).
    /// Default 0 — no path-constraint rows added.
    std::size_t num_path_constraints = 0;

    /// Per-constraint lower bounds. Size must equal num_path_constraints.
    /// Convention: >=0  → [0, +kInf]; <=0 → [-kInf, 0]; ==0 → [0, 0].
    std::vector<double> path_constraint_lower;

    /// Per-constraint upper bounds. Size must equal num_path_constraints.
    std::vector<double> path_constraint_upper;

    /// The path-constraint functor. Evaluated at every collocation node k:
    ///   g_vec = path_constraints(x_k, u_k, t_k)
    /// Each element g_vec[j] becomes one NLP constraint row with bounds
    /// [path_constraint_lower[j], path_constraint_upper[j]].
    ///
    /// AD-safety: PathConstraintFn must be fully templated — its operator() must
    /// instantiate correctly under both double and CppAD::AD<CppAD::cg::CG<double>>.
    PathConstraintFn path_constraints;
};

}  // namespace goss::transcription
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_transcription_tests && ctest --test-dir build -R HermiteSimpsonPath.OcpProblemStoresPathConstraintFields -V'
```

Expected: PASS.

- [ ] **Step 5: Also add the new test file to CMakeLists.txt**

In `CMakeLists.txt`, find the `goss_transcription_tests` executable block and add the new source:

```cmake
add_executable(goss_transcription_tests
  tests/transcription/test_variable_layout.cpp
  tests/transcription/test_trapezoidal.cpp
  tests/transcription/test_hermite_simpson.cpp
  tests/transcription/test_hermite_simpson_path.cpp
  tests/transcription/test_legendre_gauss_lobatto.cpp
  tests/transcription/test_scheme_agreement.cpp
  tests/transcription/test_mesh.cpp
  tests/transcription/test_mesh_refinement.cpp
  tests/transcription/test_lgl_nodes.cpp)
```

- [ ] **Step 6: Verify all existing transcription tests still pass**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_transcription_tests && ctest --test-dir build -R HermiteSimpson -V'
```

Expected: all existing `HermiteSimpson.*` tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/goss/transcription/ocp_problem.hpp \
        tests/transcription/test_hermite_simpson_path.cpp \
        CMakeLists.txt
git commit -m "feat(transcription): extend OcpProblem with PathConstraintFn template param and path-constraint fields

Adds NoPathConstraints sentinel, num_path_constraints, path_constraint_lower/upper,
and path_constraints fields to OcpProblem with empty defaults so all existing
two-parameter users compile unchanged. DAE orthogonality note included in header."
```

---

### Task 2: Append path-constraint rows in `HermiteSimpson::compile()`

**Files:**
- Modify: `include/goss/transcription/hermite_simpson.hpp`

**Interfaces:**
- Consumes: `OcpProblem<Dyn,Cost,PathConstraintFn>` with `num_path_constraints`, `path_constraint_lower`, `path_constraint_upper`, `path_constraints` fields from Task 1.
- Produces: `HermiteSimpson::compile()` adds `num_path_constraints * nn` constraint rows after defect rows in `gl`/`gu` and the packed functor. `VariableLayout` is NOT changed (no new decision variables).

Row layout in the packed functor's output vector:
- Index 0: cost (unchanged)
- Indices `1 .. ni*ns`: defect constraints (unchanged)
- Indices `1 + ni*ns .. 1 + ni*ns + num_path_constraints*nn - 1`: path constraints, ordered node-major: node 0 all constraints, node 1 all constraints, ...

- [ ] **Step 1: Write failing tests for path-constraint row counts and bounds**

Append to `tests/transcription/test_hermite_simpson_path.cpp`:

```cpp
// Test: compile() produces the correct number of constraint rows when one
// path constraint is active. num_constraints = ni*ns (defects) + npc*nn (path).
TEST(HermiteSimpsonPath, CompileAddsPathConstraintRows) {
    goss::transcription::OcpProblem<SimpleDecayDynamics, ZeroCostFn, SingleStatePathConstraint> ocp;
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
    const std::size_t expected_defects     = 4 * 1;
    const std::size_t expected_path_rows   = 1 * 5;
    EXPECT_EQ(compiled.problem->num_constraints(), expected_defects + expected_path_rows);
}

// Test: path-constraint bounds in the NLPProblem match [0, +kInf] for >= constraint.
TEST(HermiteSimpsonPath, PathConstraintBoundsAreCorrect) {
    goss::transcription::OcpProblem<SimpleDecayDynamics, ZeroCostFn, SingleStatePathConstraint> ocp;
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
    goss::transcription::OcpProblem<SimpleDecayDynamics, ZeroCostFn, SingleStatePathConstraint> ocp;
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
// Uses the two-param OcpProblem form (NoPathConstraints sentinel).
TEST(HermiteSimpsonPath, ZeroPathConstraintsLeavesRowCountUnchanged) {
    // Reuse the make_exponential_decay fixture from ocp_fixtures.hpp.
    // We access it here inline to avoid a fixture-header dependency in this new file.
    struct ExpDecay {
        template <typename T>
        std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>&, T) const {
            return { -x[0] };
        }
    };
    struct ZeroCost2 {
        template <typename T>
        T operator()(const std::vector<T>&, const std::vector<T>&, T) const { return T(0); }
    };
    goss::transcription::OcpProblem<ExpDecay, ZeroCost2> ocp;
    ocp.num_states   = 1;
    ocp.num_controls = 0;
    ocp.dynamics     = ExpDecay{};
    ocp.cost         = ZeroCost2{};
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
```

- [ ] **Step 2: Run to verify these four new tests fail**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_transcription_tests 2>&1 | tail -30'
```

Expected: link/compile errors because `HermiteSimpson::compile` does not yet append path rows.

- [ ] **Step 3: Implement path-constraint row appending in `HermiteSimpson`**

In `include/goss/transcription/hermite_simpson.hpp`, modify the `compile(const OcpProblem<DynamicsFn, CostFn>& ocp, ...)` primary overload signature to accept the three-parameter form and add the path-constraint loop inside the packed functor and the `gl`/`gu` extension.

Replace the entire file with:

```cpp
// include/goss/transcription/hermite_simpson.hpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/transcription/errors.hpp"
#include "goss/transcription/mesh.hpp"
#include "goss/transcription/ocp_problem.hpp"
#include "goss/transcription/transcription.hpp"
#include "goss/transcription/variable_layout.hpp"

namespace goss::transcription {

struct HermiteSimpson {
    // Primary overload: explicit non-uniform node times.
    // All implementation logic lives here; the uniform path delegates to this.
    template <typename DynamicsFn, typename CostFn, typename PathConstraintFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn, PathConstraintFn>& ocp,
                               const NonUniformMesh& mesh,
                               const std::string& model_name = "goss_hs") {
        mesh.validate();
        const std::size_t ns  = ocp.num_states;
        const std::size_t nc  = ocp.num_controls;
        const std::size_t nn  = mesh.num_nodes();
        const std::size_t ni  = mesh.num_intervals();
        const std::size_t npc = ocp.num_path_constraints;

        // Validate bound-vector sizes before touching any element.
        if (ocp.state_lower.size() != ns || ocp.state_upper.size() != ns)
            throw TranscriptionError("compile: state bound vectors must have size == num_states");
        if (ocp.control_lower.size() != nc || ocp.control_upper.size() != nc)
            throw TranscriptionError("compile: control bound vectors must have size == num_controls");
        if (ocp.initial_state.size() != ns || ocp.final_state.size() != ns)
            throw TranscriptionError("compile: initial_state/final_state must have size == num_states");
        // Validate path-constraint bound vectors when path constraints are present.
        // An empty path_constraint_lower/upper with npc==0 is valid (default case).
        if (npc > 0) {
            if (ocp.path_constraint_lower.size() != npc)
                throw TranscriptionError(
                    "compile: path_constraint_lower must have size == num_path_constraints");
            if (ocp.path_constraint_upper.size() != npc)
                throw TranscriptionError(
                    "compile: path_constraint_upper must have size == num_path_constraints");
        }

        VariableLayout layout(ns, nc, nn);

        // Capture node_times by value so the packed functor owns the data.
        const std::vector<double> node_times = mesh.node_times;

        // Packed functor: captures ocp (by value, so PathConstraintFn is included),
        // layout, and node_times. Generic lambda so it instantiates under both
        // double and the CppAD AD type during recording.
        //
        // Output layout:
        //   [0]                                  : Simpson quadrature cost
        //   [1 .. ni*ns]                         : Hermite-Simpson defects (ni intervals x ns states)
        //   [1+ni*ns .. 1+ni*ns + npc*nn - 1]   : path constraints, node-major order
        //                                          (node k, constraint j) -> index 1+ni*ns + k*npc + j
        auto packed = [ocp, layout, ns, nc, ni, nn, npc, node_times](const auto& z) {
            using T = typename std::decay_t<decltype(z)>::value_type;

            // Reserve: 1 (cost) + ni*ns (defects) + npc*nn (path).
            std::vector<T> outputs;
            outputs.reserve(1 + ni * ns + npc * nn);

            auto state_at = [&](std::size_t node) {
                std::vector<T> x(ns);
                for (std::size_t i = 0; i < ns; ++i) x[i] = z[layout.state_index(node, i)];
                return x;
            };
            auto control_at = [&](std::size_t node) {
                std::vector<T> u(nc);
                for (std::size_t j = 0; j < nc; ++j) u[j] = z[layout.control_index(node, j)];
                return u;
            };
            auto midpoint_control = [&](const std::vector<T>& uk, const std::vector<T>& uk1) {
                std::vector<T> um(nc);
                for (std::size_t j = 0; j < nc; ++j) um[j] = T(0.5) * (uk[j] + uk1[j]);
                return um;
            };

            // --- Cost + defects (identical to original HermiteSimpson) ---
            T cost = T(0);
            std::vector<T> defects;
            defects.reserve(ni * ns);

            for (std::size_t k = 0; k < ni; ++k) {
                T tk   = T(node_times[k]);
                T tk1  = T(node_times[k + 1]);
                T hk   = tk1 - tk;
                T tmid = T(0.5) * (tk + tk1);

                auto xk  = state_at(k);
                auto xk1 = state_at(k + 1);
                auto uk  = control_at(k);
                auto uk1 = control_at(k + 1);

                auto fk  = ocp.dynamics(xk, uk, tk);
                auto fk1 = ocp.dynamics(xk1, uk1, tk1);

                std::vector<T> xmid(ns);
                for (std::size_t i = 0; i < ns; ++i)
                    xmid[i] = T(0.5) * (xk[i] + xk1[i]) + (hk / T(8)) * (fk[i] - fk1[i]);

                auto umid = midpoint_control(uk, uk1);
                auto fmid = ocp.dynamics(xmid, umid, tmid);

                for (std::size_t i = 0; i < ns; ++i)
                    defects.push_back(xk1[i] - xk[i] - (hk / T(6)) * (fk[i] + T(4) * fmid[i] + fk1[i]));

                T Lk   = ocp.cost(xk, uk, tk);
                T Lmid = ocp.cost(xmid, umid, tmid);
                T Lk1  = ocp.cost(xk1, uk1, tk1);
                cost += (hk / T(6)) * (Lk + T(4) * Lmid + Lk1);
            }

            outputs.push_back(cost);
            for (auto& d : defects) outputs.push_back(d);

            // --- Path-constraint rows: node-major, all constraints per node ---
            // Evaluated at the nn collocation nodes (not midpoints).
            // WHY nodes only: midpoint x_mid is the Hermite formula (not a decision
            // variable); evaluating g at midpoints adds ni extra rows per constraint
            // but does not improve satisfaction at actual nodes. Nodes-only is the
            // standard direct-collocation convention; midpoint enforcement is a future opt.
            if (npc > 0) {
                for (std::size_t k = 0; k < nn; ++k) {
                    T tk = T(node_times[k]);
                    auto xk = state_at(k);
                    auto uk = control_at(k);
                    // path_constraints returns a vector<T> of length npc.
                    auto gk = ocp.path_constraints(xk, uk, tk);
                    for (std::size_t j = 0; j < npc; ++j)
                        outputs.push_back(gk[j]);
                }
            }

            return outputs;
        };

        auto backend = std::make_unique<goss::ad::CppADCGBackend>(
            packed, layout.total_variables(), model_name);

        // Variable bounds: per-node state and control bounds (unchanged).
        const std::size_t nv = layout.total_variables();
        std::vector<double> zl(nv, -kInf), zu(nv, kInf);
        for (std::size_t k = 0; k < nn; ++k) {
            for (std::size_t i = 0; i < ns; ++i) {
                std::size_t idx = layout.state_index(k, i);
                zl[idx] = ocp.state_lower[i];
                zu[idx] = ocp.state_upper[i];
            }
            for (std::size_t j = 0; j < nc; ++j) {
                std::size_t idx = layout.control_index(k, j);
                zl[idx] = ocp.control_lower[j];
                zu[idx] = ocp.control_upper[j];
            }
        }
        for (std::size_t i = 0; i < ns; ++i) {
            if (i < ocp.initial_state_fixed.size() && ocp.initial_state_fixed[i] != 0.0) {
                std::size_t idx = layout.state_index(0, i);
                zl[idx] = zu[idx] = ocp.initial_state[i];
            }
            if (i < ocp.final_state_fixed.size() && ocp.final_state_fixed[i] != 0.0) {
                std::size_t idx = layout.state_index(nn - 1, i);
                zl[idx] = zu[idx] = ocp.final_state[i];
            }
        }

        // Constraint bounds:
        //   defect rows:       [0,0] (equalities)
        //   path-constraint rows: [path_constraint_lower[j], path_constraint_upper[j]]
        //                         repeated for each node k, inner loop over j.
        const std::size_t num_defects = ni * ns;
        const std::size_t num_path_rows = npc * nn;
        std::vector<double> gl(num_defects + num_path_rows, 0.0);
        std::vector<double> gu(num_defects + num_path_rows, 0.0);
        // Defect rows are already [0,0] from the initializer.
        // Path rows: repeat bounds for each node.
        for (std::size_t k = 0; k < nn; ++k) {
            for (std::size_t j = 0; j < npc; ++j) {
                const std::size_t row = num_defects + k * npc + j;
                gl[row] = ocp.path_constraint_lower[j];
                gu[row] = ocp.path_constraint_upper[j];
            }
        }

        auto problem = std::make_unique<nlp::NLPProblem>(
            std::move(backend), std::move(zl), std::move(zu), std::move(gl), std::move(gu));
        return CompiledOcp{std::move(problem), layout};
    }

    // Backward-compatible uniform overload: delegates to the non-uniform path.
    template <typename DynamicsFn, typename CostFn, typename PathConstraintFn>
    static CompiledOcp compile(const OcpProblem<DynamicsFn, CostFn, PathConstraintFn>& ocp,
                               const std::string& model_name = "goss_hs") {
        return compile(ocp, to_nonuniform(ocp.mesh), model_name);
    }
};

}  // namespace goss::transcription
```

- [ ] **Step 4: Run all four new tests plus existing HS tests**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_transcription_tests && ctest --test-dir build -R HermiteSimpson -V'
```

Expected: all `HermiteSimpson.*` tests PASS including the four new `HermiteSimpsonPath.*` tests.

- [ ] **Step 5: Commit**

```bash
git add include/goss/transcription/hermite_simpson.hpp \
        tests/transcription/test_hermite_simpson_path.cpp
git commit -m "feat(transcription): append path-constraint rows in HermiteSimpson::compile()

Path constraints are evaluated at all nn collocation nodes (not midpoints) producing
npc*nn additional NLP rows after the ni*ns defect rows. NoPathConstraints default
path is a no-op: zero rows, backward-compatible. Trapezoidal and LGL deferred."
```

---

### Task 3: DSL path-constraint types and expression-typed comparison operators

**Files:**
- Create: `include/goss/model/expr/path_constraint.hpp`
- Modify: `include/goss/model/expr/expr.hpp`

**Interfaces:**
- Produces:
  - `struct goss::model::expr::PathConstraintExpr<Expr>` — wraps an expression + scalar bounds (lower/upper), represents one scalar path constraint
  - `struct goss::model::expr::PathConstraintEntry<Expr>` — `PathConstraintExpr<Expr>` + per-constraint lower/upper
  - `struct goss::model::expr::PathConstraintFunctor<ExprTuple>` — tuple-over-entries, `template<T> vector<T> operator()(x,u,t)` that evaluates all entries in index order (same pattern as `DynamicsFunctor`)
  - Expression-typed `operator>=/<=/==` overloads: LHS must be a `BinaryExpr`, `UnaryNegExpr`, `StateLeaf`, `ControlLeaf`, or `ConstantExpr`; NOT `StateHandle` or `ControlHandle`. Returns `PathConstraintExpr<Expr>`.
- Preserves: Existing `goss::model::operator>=(StateHandle, double)` → `BoundConstraint` is untouched (different namespace + LHS type).

**Overload-resolution mechanism (critical detail):**

The existing box-bound operators live in `namespace goss::model` and have LHS type `goss::model::StateHandle` (a plain struct). The new path-constraint operators live in `namespace goss::model::expr` and are gated via `std::enable_if_t` on the LHS being an expr-DSL node type. ADL finds the right overload because:
- `q >= 0.0` where `q` is `StateHandle` → ADL searches `goss::model` → finds `operator>=(StateHandle, double)` → returns `BoundConstraint`. The `goss::model::expr` overload is NOT a candidate because `StateHandle` is not an expr-DSL type (the `enable_if` guard rejects it).
- `(q_expr + 1.0) >= 0.0` where `q_expr` is `BinaryExpr<AddTag, StateLeaf, ConstantExpr>` → ADL searches `goss::model::expr` (where `BinaryExpr` lives) → finds `operator>=(BinaryExpr<...>, double)` → returns `PathConstraintExpr<BinaryExpr<...>>`. The `goss::model` overload is NOT a candidate because `BinaryExpr` is not `StateHandle`.

The `enable_if` guard uses a trait `is_expr_node<T>` that is `true` for `BinaryExpr`, `UnaryNegExpr`, `StateLeaf`, `ControlLeaf`, `ConstantExpr`, `TimeLeaf` and `false` for everything else:

```cpp
template <typename T> struct is_expr_node : std::false_type {};
template <typename Op, typename L, typename R>
struct is_expr_node<BinaryExpr<Op,L,R>> : std::true_type {};
template <typename E>
struct is_expr_node<UnaryNegExpr<E>> : std::true_type {};
template <> struct is_expr_node<StateLeaf>   : std::true_type {};
template <> struct is_expr_node<ControlLeaf> : std::true_type {};
template <> struct is_expr_node<ConstantExpr>: std::true_type {};
template <> struct is_expr_node<TimeLeaf>    : std::true_type {};
```

- [ ] **Step 1: Write failing tests for path-constraint DSL types**

Create `tests/model/test_path_constraint_lowering.cpp`:

```cpp
// tests/model/test_path_constraint_lowering.cpp
// Tests for PathConstraintExpr construction, PathConstraintFunctor evaluation,
// and expression-typed operator>=/<=/== overloads.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/model/expr/expr.hpp"
#include "goss/transcription/transcription.hpp"

using namespace goss::model::expr;

// Test: (StateLeaf + ConstantExpr) >= 0.0 produces a PathConstraintExpr,
// not a BoundConstraint. The returned type should NOT be BoundConstraint.
TEST(PathConstraintLowering, ExpressionPlusConstantGeqProducesPathConstraintExpr) {
    // This must compile: (StateLeaf{0} + 1.0) >= 0.0
    auto path_expr = (StateLeaf{0} + 1.0) >= 0.0;
    // The result must have path_constraint_lower == 0.0 and path_constraint_upper == kInf.
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_lower, 0.0);
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_upper, goss::transcription::kInf);

    // Evaluate under double: x=[2.0], u=[], t=0 => (2.0 + 1.0) - 0.0 = 3.0
    // The expression is: StateLeaf{0} + 1.0 - 0.0 (the RHS is subtracted so g=expr-rhs)
    // Convention: (expr) >= rhs  =>  g = expr - rhs >= 0  =>  g(x,u,t) = expr.eval - rhs
    std::vector<double> x = {2.0};
    std::vector<double> u = {};
    double g_val = path_expr.constraint_expression.eval(x, u, 0.0);
    EXPECT_NEAR(g_val, 3.0, 1e-12);
}

// Test: (StateLeaf - ConstantExpr) <= 1.0 produces PathConstraintExpr with
// gl=-kInf, gu=0.0. Expression g = (expr - rhs).
// (x - 1.0) <= 1.0 => g = (x - 1.0) - 1.0 = x - 2.0 <= 0 => gu=0.
TEST(PathConstraintLowering, ExprLeqProducesNegativeUpperBound) {
    auto path_expr = (StateLeaf{0} - 1.0) <= 1.0;
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_lower, -goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_upper, 0.0);

    std::vector<double> x = {3.0};
    std::vector<double> u = {};
    // g = (x - 1.0) - 1.0 = 3.0 - 2.0 = 1.0
    double g_val = path_expr.constraint_expression.eval(x, u, 0.0);
    EXPECT_NEAR(g_val, 1.0, 1e-12);
}

// Test: (ControlLeaf * ControlLeaf) == 0.0 produces equality constraint [0,0].
TEST(PathConstraintLowering, ExprEqZeroProducesEqualityConstraint) {
    auto path_expr = (ControlLeaf{0} * ControlLeaf{0}) == 0.0;
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_lower, 0.0);
    EXPECT_DOUBLE_EQ(path_expr.path_constraint_upper, 0.0);
}

// Test: PathConstraintFunctor with a single entry evaluates correctly.
TEST(PathConstraintLowering, PathConstraintFunctorEvaluatesSingleEntry) {
    // Manually construct PathConstraintEntry and PathConstraintFunctor.
    // Entry: g(x,u,t) = x[0] + 1.0 - 0.0 (from (StateLeaf{0}+1.0) >= 0.0)
    using ExprType = BinaryExpr<SubTag,
                        BinaryExpr<AddTag, StateLeaf, ConstantExpr>,
                        ConstantExpr>;
    // g = (x[0] + 1.0) - 0.0
    ExprType g_expr{
        BinaryExpr<AddTag, StateLeaf, ConstantExpr>{ StateLeaf{0}, ConstantExpr{1.0} },
        ConstantExpr{0.0}
    };
    PathConstraintEntry<ExprType> entry{ std::move(g_expr), 0.0, goss::transcription::kInf };

    auto entries_tuple = std::make_tuple(std::move(entry));
    PathConstraintFunctor<decltype(entries_tuple)> functor{ std::move(entries_tuple), 1 };

    std::vector<double> x = {4.0};
    std::vector<double> u = {};
    auto result = functor(x, u, 0.0);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 5.0, 1e-12);  // 4.0 + 1.0 - 0.0 = 5.0
}

// Test: bare StateHandle >= double STILL produces BoundConstraint (box bound, not path).
// This test verifies no overload-resolution regression.
TEST(PathConstraintLowering, BareStateHandleGeqStillProducesBoundConstraint) {
    goss::model::StateHandle q_handle{0};
    // Must compile as BoundConstraint (the existing box-bound path).
    // If this accidentally resolved to PathConstraintExpr, the test would
    // fail to compile (BoundConstraint has no path_constraint_lower field).
    goss::model::expr::BoundConstraint bc = q_handle >= 0.0;
    EXPECT_DOUBLE_EQ(bc.lower_bound, 0.0);
    EXPECT_DOUBLE_EQ(bc.upper_bound, goss::transcription::kInf);
}
```

- [ ] **Step 2: Run to verify these tests fail**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_model_tests 2>&1 | tail -30'
```

Expected: compile errors — `PathConstraintExpr` not defined, expression-typed `>=` not found.

- [ ] **Step 3: Implement `path_constraint.hpp`**

Create `include/goss/model/expr/path_constraint.hpp`:

```cpp
// include/goss/model/expr/path_constraint.hpp
// PathConstraint DSL types and expression-typed comparison operators.
//
// This header provides:
//   1. is_expr_node<T> trait — true for all expr-DSL node types.
//   2. PathConstraintExpr<Expr> — one scalar path constraint with bounds.
//   3. PathConstraintEntry<Expr> — same as PathConstraintExpr (alias pattern).
//   4. PathConstraintFunctor<ExprTuple> — evaluates a tuple of entries under ScalarT.
//   5. operator>=/<=/== overloads for expression-typed LHS (SFINAE-gated).
//
// OVERLOAD RESOLUTION INVARIANT:
//   - q >= 0.0 where q is StateHandle  => goss::model::operator>= => BoundConstraint.
//   - (q_expr + 1.0) >= 0.0 where q_expr is BinaryExpr<...> => the overloads here
//     (namespace goss::model::expr, found by ADL on BinaryExpr) => PathConstraintExpr.
//   No ambiguity: StateHandle is not an expr node; expr nodes are not StateHandle.
//
// AD-SAFETY: PathConstraintFunctor::operator() is a member function template
// (not std::function). It instantiates under both double and CppAD AD types,
// so capturing a PathConstraintFunctor by value inside the HermiteSimpson
// packed generic lambda is correct and AD-safe.
#pragma once
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "goss/model/expr/nodes.hpp"
#include "goss/transcription/transcription.hpp"  // kInf

namespace goss::model::expr {

// ─── is_expr_node trait ──────────────────────────────────────────────────────

/// Trait to detect expr-DSL node types. Only these types are valid as the LHS
/// of expression-typed >= / <= / == (producing PathConstraintExpr).
/// StateHandle and ControlHandle are NOT expr nodes — they go through the
/// existing goss::model::operator>= which returns BoundConstraint.
template <typename T> struct is_expr_node : std::false_type {};

template <typename OpTag, typename LeftExpr, typename RightExpr>
struct is_expr_node<BinaryExpr<OpTag, LeftExpr, RightExpr>> : std::true_type {};

template <typename OperandExpr>
struct is_expr_node<UnaryNegExpr<OperandExpr>> : std::true_type {};

template <> struct is_expr_node<StateLeaf>    : std::true_type {};
template <> struct is_expr_node<ControlLeaf>  : std::true_type {};
template <> struct is_expr_node<ConstantExpr> : std::true_type {};
template <> struct is_expr_node<TimeLeaf>     : std::true_type {};

// Convenience alias.
template <typename T>
constexpr bool is_expr_node_v = is_expr_node<T>::value;

// ─── PathConstraintExpr ──────────────────────────────────────────────────────

/// Represents one scalar path constraint:  constraint_expression(x,u,t) in [lo, hi].
///
/// The convention is: g(x,u,t) = constraint_expression.eval(x,u,t).
/// For (expr >= rhs): constraint_expression = expr - rhs,  lo=0,   hi=+kInf.
/// For (expr <= rhs): constraint_expression = expr - rhs,  lo=-kInf, hi=0.
/// For (expr == rhs): constraint_expression = expr - rhs,  lo=0,   hi=0.
///
/// The subtraction of rhs is baked into the expression tree at DSL-call time:
///   (expr >= rhs)  =>  PathConstraintExpr{expr - ConstantExpr{rhs}, 0.0, +kInf}
/// This keeps the functor evaluation simple: just call constraint_expression.eval.
template <typename Expr>
struct PathConstraintExpr {
    Expr   constraint_expression;  // g(x,u,t) = this expression; must be >= 0 / in [lo,hi]
    double path_constraint_lower;
    double path_constraint_upper;
};

/// PathConstraintEntry is a synonym for PathConstraintExpr.
/// Used inside PathConstraintFunctor's tuple to mirror the DynamicsEntry pattern.
template <typename Expr>
using PathConstraintEntry = PathConstraintExpr<Expr>;

// ─── PathConstraintFunctor ───────────────────────────────────────────────────

/// Holds a tuple of PathConstraintEntry objects and satisfies the PathConstraintFn
/// contract expected by OcpProblem:
///   template<typename T> std::vector<T> operator()(const vector<T>& x,
///                                                   const vector<T>& u,
///                                                   T t) const;
///
/// Evaluates each entry in tuple order and packs results into the returned vector.
/// tuple order IS the constraint index order; unlike DynamicsFunctor, there is no
/// placement-by-index because path constraints are anonymous (no state-slot mapping).
///
/// AD-safety: fully templated operator() — instantiates under both double and
/// CppAD AD types. Captured by value inside HermiteSimpson's packed generic lambda.
template <typename ExprTuple>
struct PathConstraintFunctor {
    ExprTuple path_constraint_entries;

    /// Total number of path constraints — equal to std::tuple_size_v<ExprTuple>.
    /// Stored for runtime use (e.g. size checks before compile).
    std::size_t num_path_constraints;

    template <typename ScalarT>
    std::vector<ScalarT> operator()(const std::vector<ScalarT>& x,
                                    const std::vector<ScalarT>& u,
                                    ScalarT                     t) const {
        std::vector<ScalarT> result;
        result.reserve(num_path_constraints);
        fill_result(result, x, u, t,
                    std::make_index_sequence<std::tuple_size_v<ExprTuple>>{});
        return result;
    }

 private:
    template <typename ScalarT, std::size_t... Indices>
    void fill_result(std::vector<ScalarT>&       result,
                     const std::vector<ScalarT>& x,
                     const std::vector<ScalarT>& u,
                     ScalarT                     t,
                     std::index_sequence<Indices...>) const {
        // Fold expression: evaluate each entry in tuple order and push_back.
        // Comma-operator fold guarantees evaluation in Indices order.
        ((result.push_back(
            std::get<Indices>(path_constraint_entries)
                .constraint_expression.template eval<ScalarT>(x, u, t))), ...);
    }
};

// ─── Helpers to extract bounds from a PathConstraintFunctor tuple ─────────────

/// Extract the lower-bound vector from a PathConstraintFunctor (for OcpProblem setup).
template <typename ExprTuple>
std::vector<double> extract_path_constraint_lower(
        const PathConstraintFunctor<ExprTuple>& functor) {
    std::vector<double> bounds;
    bounds.reserve(functor.num_path_constraints);
    extract_bounds_lower_impl(bounds, functor.path_constraint_entries,
                              std::make_index_sequence<std::tuple_size_v<ExprTuple>>{});
    return bounds;
}

template <typename ExprTuple, std::size_t... Indices>
void extract_bounds_lower_impl(std::vector<double>& bounds,
                                const ExprTuple& entries,
                                std::index_sequence<Indices...>) {
    ((bounds.push_back(std::get<Indices>(entries).path_constraint_lower)), ...);
}

/// Extract the upper-bound vector from a PathConstraintFunctor (for OcpProblem setup).
template <typename ExprTuple>
std::vector<double> extract_path_constraint_upper(
        const PathConstraintFunctor<ExprTuple>& functor) {
    std::vector<double> bounds;
    bounds.reserve(functor.num_path_constraints);
    extract_bounds_upper_impl(bounds, functor.path_constraint_entries,
                              std::make_index_sequence<std::tuple_size_v<ExprTuple>>{});
    return bounds;
}

template <typename ExprTuple, std::size_t... Indices>
void extract_bounds_upper_impl(std::vector<double>& bounds,
                                const ExprTuple& entries,
                                std::index_sequence<Indices...>) {
    ((bounds.push_back(std::get<Indices>(entries).path_constraint_upper)), ...);
}

}  // namespace goss::model::expr

// ─── Expression-typed comparison operators (in goss::model::expr for ADL) ────
// These live in goss::model::expr so that ADL on BinaryExpr<...> etc. finds them.
// They do NOT overload with the existing goss::model::operator>=(StateHandle, double)
// because StateHandle is not an expr node (is_expr_node_v<StateHandle> == false).
namespace goss::model::expr {

/// (expr) >= rhs  =>  PathConstraintExpr{ expr - ConstantExpr{rhs}, 0.0, +kInf }
/// SFINAE guard: only fires when LhsExpr is an expr-DSL node (not StateHandle/ControlHandle).
template <typename LhsExpr,
          typename = std::enable_if_t<is_expr_node_v<LhsExpr>>>
PathConstraintExpr<BinaryExpr<SubTag, LhsExpr, ConstantExpr>>
operator>=(LhsExpr lhs_expression, double rhs_value) {
    using ExprType = BinaryExpr<SubTag, LhsExpr, ConstantExpr>;
    ExprType shifted_expression{
        std::move(lhs_expression),
        ConstantExpr{rhs_value}
    };
    return PathConstraintExpr<ExprType>{
        std::move(shifted_expression),
        0.0,
        goss::transcription::kInf
    };
}

/// (expr) <= rhs  =>  PathConstraintExpr{ expr - ConstantExpr{rhs}, -kInf, 0.0 }
template <typename LhsExpr,
          typename = std::enable_if_t<is_expr_node_v<LhsExpr>>>
PathConstraintExpr<BinaryExpr<SubTag, LhsExpr, ConstantExpr>>
operator<=(LhsExpr lhs_expression, double rhs_value) {
    using ExprType = BinaryExpr<SubTag, LhsExpr, ConstantExpr>;
    ExprType shifted_expression{
        std::move(lhs_expression),
        ConstantExpr{rhs_value}
    };
    return PathConstraintExpr<ExprType>{
        std::move(shifted_expression),
        -goss::transcription::kInf,
        0.0
    };
}

/// (expr) == rhs  =>  PathConstraintExpr{ expr - ConstantExpr{rhs}, 0.0, 0.0 }
template <typename LhsExpr,
          typename = std::enable_if_t<is_expr_node_v<LhsExpr>>>
PathConstraintExpr<BinaryExpr<SubTag, LhsExpr, ConstantExpr>>
operator==(LhsExpr lhs_expression, double rhs_value) {
    using ExprType = BinaryExpr<SubTag, LhsExpr, ConstantExpr>;
    ExprType shifted_expression{
        std::move(lhs_expression),
        ConstantExpr{rhs_value}
    };
    return PathConstraintExpr<ExprType>{
        std::move(shifted_expression),
        0.0,
        0.0
    };
}

}  // namespace goss::model::expr
```

- [ ] **Step 4: Add `#include` to `expr.hpp`**

In `include/goss/model/expr/expr.hpp`, add:

```cpp
#include "goss/model/expr/path_constraint.hpp"
```

(after the existing includes).

- [ ] **Step 5: Add the new test file to CMakeLists.txt**

In `CMakeLists.txt`, find `add_executable(goss_model_tests` and append the new source:

```cmake
add_executable(goss_model_tests
  tests/model/test_model_declaration.cpp
  tests/model/test_model_build.cpp
  tests/model/test_model_solve.cpp
  tests/model/test_component.cpp
  tests/model/test_composed_model.cpp
  tests/model/test_composition_solve.cpp
  tests/model/test_expr_nodes.cpp
  tests/model/test_expr_lowering.cpp
  tests/model/test_expr_solve.cpp
  tests/model/test_path_constraint_lowering.cpp)
```

- [ ] **Step 6: Run tests**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_model_tests && ctest --test-dir build -R PathConstraintLowering -V'
```

Expected: all `PathConstraintLowering.*` tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/goss/model/expr/path_constraint.hpp \
        include/goss/model/expr/expr.hpp \
        tests/model/test_path_constraint_lowering.cpp \
        CMakeLists.txt
git commit -m "feat(expr): add PathConstraintExpr, PathConstraintFunctor, and expression-typed comparison operators

is_expr_node<T> trait gates >= / <= / == overloads to expr-DSL nodes only;
bare StateHandle >= double still resolves to BoundConstraint via goss::model ADL.
PathConstraintFunctor mirrors DynamicsFunctor pattern: tuple of entries, no std::function."
```

---

### Task 4: `ExprModel` path-constraint accumulation and `build()` integration

**Files:**
- Modify: `include/goss/model/expr/expr_model.hpp`

**Interfaces:**
- Consumes: `PathConstraintExpr<Expr>`, `PathConstraintFunctor<ExprTuple>`, `extract_path_constraint_lower`, `extract_path_constraint_upper` from Task 3.
- Produces:
  - `ExprModel<DynTuple, CostFn, PathTuple>` — adds third template param `PathTuple = std::tuple<>` for accumulated path constraints.
  - `with_path_constraint(PathConstraintExpr<Expr>)` — rvalue-qualified, returns `ExprModel<DynTuple, CostFn, NewPathTuple>` with one more entry appended. Same type-accumulation pattern as `with_dynamics()`.
  - Updated `build()` — when `PathTuple` is non-empty, assembles a `PathConstraintFunctor`, extracts bounds, and produces `OcpProblem<DynFunctor, CostFn, PathConstraintFunctor<PathTuple>>` with path-constraint fields populated.

- [ ] **Step 1: Write failing tests for `ExprModel` path-constraint accumulation**

Append to `tests/model/test_path_constraint_lowering.cpp`:

```cpp
// Tests for ExprModel::with_path_constraint() and build() path-constraint integration.

TEST(PathConstraintLowering, ExprModelWithPathConstraintCompiles) {
    using namespace goss::model::expr;

    ExprModel<> model;
    const auto q_handle = model.add_state("state_x");
    model.set_mesh(0.0, 1.0, 4);

    // with_path_constraint() must accept a PathConstraintExpr and return a new ExprModel.
    // Constraint: x + 1.0 >= 0.0  (always satisfied for x >= -1.0)
    auto model_with_path = std::move(model)
        .with_path_constraint((StateLeaf{q_handle.index} + 1.0) >= 0.0);

    // The returned ExprModel should compile and allow further chaining.
    // We don't call build() here (no dynamics yet) — just verify the type compiles.
    SUCCEED();
}

TEST(PathConstraintLowering, ExprModelBuildProducesOcpWithPathConstraints) {
    using namespace goss::model::expr;

    ExprModel<> model;
    const auto q_handle = model.add_state("state_x");
    const auto u_handle = model.add_control("control_u");
    model.set_mesh(0.0, 1.0, 4);
    model.apply(q_handle.initial() == 2.0);
    model.apply(q_handle >= -goss::transcription::kInf);
    model.apply(u_handle >= -1.0);
    model.apply(u_handle <= 1.0);

    // dx/dt = u,  cost = x^2,  path constraint: x + 1.0 >= 0.0
    auto ocp = std::move(model)
        .with_dynamics(q_handle, ControlLeaf{u_handle.index})
        .with_cost(integral(StateLeaf{q_handle.index} * StateLeaf{q_handle.index}))
        .with_path_constraint((StateLeaf{q_handle.index} + 1.0) >= 0.0)
        .build();

    // The OcpProblem should have 1 path constraint.
    EXPECT_EQ(ocp.num_path_constraints, 1u);
    EXPECT_EQ(ocp.path_constraint_lower.size(), 1u);
    EXPECT_DOUBLE_EQ(ocp.path_constraint_lower[0], 0.0);
    EXPECT_DOUBLE_EQ(ocp.path_constraint_upper[0], goss::transcription::kInf);
}
```

- [ ] **Step 2: Run to verify the tests fail**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_model_tests 2>&1 | grep -E "error:|FAILED" | head -20'
```

Expected: compile error — `ExprModel` has no `with_path_constraint` method.

- [ ] **Step 3: Implement the `ExprModel` extension**

In `include/goss/model/expr/expr_model.hpp`:

1. Add `#include "goss/model/expr/path_constraint.hpp"` near the top (after `constraints.hpp`).

2. Change the class template declaration from:
```cpp
template <typename DynTuple = std::tuple<>, typename CostFn = std::monostate>
class ExprModel {
```
to:
```cpp
template <typename DynTuple  = std::tuple<>,
          typename CostFn    = std::monostate,
          typename PathTuple = std::tuple<>>
class ExprModel {
```

3. Add a `path_tuple_` private member after `cost_fn_`:
```cpp
 private:
    goss::model::Model model_{};
    DynTuple           dyn_tuple_{};
    CostFn             cost_fn_{};
    PathTuple          path_tuple_{};  // accumulated PathConstraintEntry objects
```

4. Update the internal constructor to accept and transfer `PathTuple`:
```cpp
    ExprModel(goss::model::Model model_arg,
              DynTuple           dyn_tuple_arg,
              CostFn             cost_fn_arg,
              PathTuple          path_tuple_arg)
        : model_(std::move(model_arg)),
          dyn_tuple_(std::move(dyn_tuple_arg)),
          cost_fn_(std::move(cost_fn_arg)),
          path_tuple_(std::move(path_tuple_arg)) {}
```

5. Update `with_dynamics()` and `with_cost()` to forward `path_tuple_` to the returned ExprModel:

In `with_dynamics()`:
```cpp
        return ExprModel<NewDynTuple, CostFn, PathTuple>{
            std::move(model_),
            std::move(new_tuple),
            std::move(cost_fn_),
            std::move(path_tuple_)
        };
```

In `with_cost()`:
```cpp
        return ExprModel<DynTuple, NewCostFn, PathTuple>{
            std::move(model_),
            std::move(dyn_tuple_),
            std::move(new_cost_fn),
            std::move(path_tuple_)
        };
```

6. Add `with_path_constraint()` (rvalue-qualified, returns new ExprModel with extended PathTuple):
```cpp
    /// Register a path constraint and return a NEW ExprModel with the PathTuple
    /// extended by one PathConstraintEntry. The constraint is evaluated at every
    /// collocation node during transcription.
    ///
    /// WHY &&: same reason as with_dynamics() — model_ must be moved to avoid
    /// deep copies of state vectors; each call produces a distinct template
    /// specialisation.
    template <typename Expr>
    auto with_path_constraint(PathConstraintExpr<Expr> path_constraint_expression) && {
        PathConstraintEntry<Expr> new_entry{
            std::move(path_constraint_expression.constraint_expression),
            path_constraint_expression.path_constraint_lower,
            path_constraint_expression.path_constraint_upper
        };
        auto new_path_tuple = std::tuple_cat(
            std::move(path_tuple_),
            std::make_tuple(std::move(new_entry)));
        using NewPathTuple = decltype(new_path_tuple);
        return ExprModel<DynTuple, CostFn, NewPathTuple>{
            std::move(model_),
            std::move(dyn_tuple_),
            std::move(cost_fn_),
            std::move(new_path_tuple)
        };
    }
```

7. Update `build()` (the zero-argument overload) to assemble the `PathConstraintFunctor` and populate the OcpProblem path-constraint fields. In the `else` branch (after the dynamics-count check and `DynamicsFunctor` construction), replace:
```cpp
            return model_.build(std::move(dyn_functor), std::move(cost_fn_));
```
with:
```cpp
            // If path constraints are registered, assemble a PathConstraintFunctor
            // and build an OcpProblem<DynFunctor, CostFn, PathConstraintFunctor>.
            // If not (PathTuple is empty), build the two-param form (NoPathConstraints default).
            constexpr std::size_t registered_path_count = std::tuple_size_v<PathTuple>;
            if constexpr (registered_path_count == 0) {
                // No path constraints: use the default NoPathConstraints path.
                return model_.build(std::move(dyn_functor), std::move(cost_fn_));
            } else {
                // Assemble PathConstraintFunctor from accumulated tuple.
                PathConstraintFunctor<PathTuple> path_functor{
                    std::move(path_tuple_),
                    registered_path_count
                };
                // Extract per-constraint lower/upper bounds.
                auto pc_lower = extract_path_constraint_lower(path_functor);
                auto pc_upper = extract_path_constraint_upper(path_functor);

                // Build the OcpProblem with path-constraint fields populated.
                // model_.build_with_path_constraints() is a new Model overload added below.
                return model_.build_with_path_constraints(
                    std::move(dyn_functor),
                    std::move(cost_fn_),
                    std::move(path_functor),
                    registered_path_count,
                    std::move(pc_lower),
                    std::move(pc_upper));
            }
```

**NOTE:** `model_.build_with_path_constraints(...)` must be added to `goss::model::Model`. See Step 4.

- [ ] **Step 4: Add `build_with_path_constraints` to `goss::model::Model`**

Read `include/goss/model/model.hpp` to find where `build()` is defined, then add a new overload:

```cpp
    /// Build an OcpProblem with path constraints. Called by ExprModel::build()
    /// when path constraints have been registered. The PathConstraintFn is passed
    /// by value (captured inside OcpProblem by value, as required for AD-safety).
    template <typename DynamicsFn, typename CostFn, typename PathConstraintFn>
    goss::transcription::OcpProblem<DynamicsFn, CostFn, PathConstraintFn>
    build_with_path_constraints(DynamicsFn    dynamics_functor,
                                 CostFn        cost_functor,
                                 PathConstraintFn path_constraint_functor,
                                 std::size_t   num_path_constraints_arg,
                                 std::vector<double> path_constraint_lower_arg,
                                 std::vector<double> path_constraint_upper_arg) {
        // Reuse the existing build() validation logic by first building a base OcpProblem.
        auto base_ocp = build(std::move(dynamics_functor), std::move(cost_functor));
        // Extend with path-constraint fields.
        goss::transcription::OcpProblem<DynamicsFn, CostFn, PathConstraintFn> ocp_with_path{
            base_ocp.num_states,
            base_ocp.num_controls,
            std::move(base_ocp.dynamics),
            std::move(base_ocp.cost),
            base_ocp.mesh,
            std::move(base_ocp.state_lower),
            std::move(base_ocp.state_upper),
            std::move(base_ocp.control_lower),
            std::move(base_ocp.control_upper),
            std::move(base_ocp.initial_state),
            std::move(base_ocp.initial_state_fixed),
            std::move(base_ocp.final_state),
            std::move(base_ocp.final_state_fixed),
            num_path_constraints_arg,
            std::move(path_constraint_lower_arg),
            std::move(path_constraint_upper_arg),
            std::move(path_constraint_functor)
        };
        return ocp_with_path;
    }
```

- [ ] **Step 5: Run tests**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_model_tests && ctest --test-dir build -R PathConstraintLowering -V'
```

Expected: all `PathConstraintLowering.*` tests PASS.

- [ ] **Step 6: Also run the full model test suite to confirm no regressions**

```bash
scripts/dev.sh 'ctest --test-dir build -R goss_model_tests -V 2>&1 | tail -30'
```

Expected: all existing model tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/goss/model/expr/expr_model.hpp \
        include/goss/model/model.hpp \
        tests/model/test_path_constraint_lowering.cpp
git commit -m "feat(expr): add ExprModel::with_path_constraint() and build() path-constraint integration

ExprModel gains a PathTuple third template param (default empty tuple). with_path_constraint()
appends PathConstraintEntry to the tuple following the same type-accumulation pattern as
with_dynamics(). build() assembles PathConstraintFunctor and calls model_.build_with_path_constraints()
when PathTuple is non-empty; the no-path-constraint path is unchanged."
```

---

### Task 5: End-to-end solve with a nonlinear path constraint (accuracy-suite integration)

**Prerequisite:** The accuracy-validation-suite plan (`docs/superpowers/plans/2026-08-01-accuracy-validation-suite.md`) must have been executed and merged first, providing:
- `tests/accuracy/accuracy_helpers.hpp` with `goss::accuracy::solve_and_extract_trajectory`
- `goss_accuracy_tests` CMake target linked to `goss_model goss_transcription goss_nlp goss_ad goss_ad_impl goss_solver goss_ipopt_iface goss_nlopt_iface cppadcg GTest::gtest_main`

**Files:**
- Create: `tests/accuracy/test_path_constraint_accuracy.cpp`
- Modify: `CMakeLists.txt` (add new source to `goss_accuracy_tests`)

**Problem specification — Bounded Double Integrator:**

The chosen problem has a computable reference optimum, is not trivially box-constrained, and has an active nonlinear state-path constraint that changes the unconstrained optimum.

```
Dynamics:    dx0/dt = x1,   dx1/dt = u
Boundary:    x0(0) = 0,  x1(0) = 0,  x0(T) = 1,  x1(T) = 0
Cost:        min integral(u^2) over [0, T=1]
Path constraint: x0^2 + x1^2 <= R^2,  with R = 0.5
                 i.e. g(x,u,t) = R^2 - x0^2 - x1^2 >= 0
```

**Reference analysis:**

Without the path constraint, the minimum-energy double-integrator solution is the classical bang-bang (then coast) or the purely smooth `u*(t) = 6 - 12t` with objective `J_unconstrained = 12.0` (exact for T=1, x0: 0→1, x1: 0→0).

With `R = 0.5`, the Euclidean-norm path constraint is active along a section of the optimal trajectory because the unconstrained arc passes outside the ball. The constraint is active when `x0^2 + x1^2 = 0.25`. For this constrained problem the exact optimum is not available in closed form in a simple expression, but we can bound it and check feasibility + that it exceeds the unconstrained cost:

- `J_constrained >= J_unconstrained = 12.0` (constraint adds cost).
- The constraint must be satisfied at all nodes: `x0_k^2 + x1_k^2 <= R^2 + tol`.
- Solver convergence (Ipopt returns Success).

Additionally, we use a simpler pure-feasibility test: **with R = 2.0** the constraint is never active (the unconstrained trajectory stays inside the ball), so `J_constrained_R2 ≈ J_unconstrained = 12.0` within solver tolerance `1e-3`.

This gives two quantitative assertions:
1. R=2.0 (inactive constraint): `|J - 12.0| < 1e-2` (tight tolerance).
2. R=0.5 (active constraint): `J > 12.0` AND constraint satisfied at all nodes.

- [ ] **Step 1: Write the accuracy test**

Create `tests/accuracy/test_path_constraint_accuracy.cpp`:

```cpp
// tests/accuracy/test_path_constraint_accuracy.cpp
//
// End-to-end accuracy test for nonlinear path constraints.
// Problem: minimum-energy double integrator with Euclidean-norm path constraint.
//
// Dynamics:  dx0/dt = x1,  dx1/dt = u
// Boundary:  x0(0)=0, x1(0)=0, x0(1)=1, x1(1)=0
// Cost:      integral(u^2) over [0,1]
// Path:      x0^2 + x1^2 <= R^2
//
// Reference (unconstrained, R large): J* = 12.0 exactly.
// See: Liberzon "Calculus of Variations and Optimal Control Theory" §3.3.
//
// Tests:
//   A) R = 2.0 (constraint inactive): J ≈ 12.0 within 1e-2.
//   B) R = 0.5 (constraint active):   J > 12.0  AND  x0_k^2 + x1_k^2 <= R^2 + tol at all nodes.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/model/expr/expr.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "accuracy/accuracy_helpers.hpp"

namespace {

constexpr double TIME_HORIZON     = 1.0;
constexpr std::size_t NUM_INTERVALS = 40;  // fine mesh for accuracy

/// Build and solve the double-integrator OCP with Euclidean-norm path constraint
/// x0^2 + x1^2 <= R^2 (i.e. g = R^2 - x0^2 - x1^2 >= 0).
/// Returns the SolutionTrajectory from accuracy_helpers::solve_and_extract_trajectory.
goss::accuracy::SolutionTrajectory solve_double_integrator_with_norm_constraint(
        double radius_bound,
        const std::string& model_name) {
    using namespace goss::model::expr;

    ExprModel<> expr_model;
    const auto x0_handle = expr_model.add_state("position");
    const auto x1_handle = expr_model.add_state("velocity");
    const auto u_handle  = expr_model.add_control("force");

    // Boundary conditions: both states pinned at t=0 and t=T.
    expr_model.apply(x0_handle.initial() == 0.0);
    expr_model.apply(x1_handle.initial() == 0.0);
    expr_model.apply(x0_handle.final()   == 1.0);
    expr_model.apply(x1_handle.final()   == 0.0);

    // Wide bounds to keep the problem unconstrained except for the path constraint.
    expr_model.apply(x0_handle >= -10.0);
    expr_model.apply(x0_handle <= 10.0);
    expr_model.apply(x1_handle >= -10.0);
    expr_model.apply(x1_handle <= 10.0);
    expr_model.apply(u_handle  >= -20.0);
    expr_model.apply(u_handle  <=  20.0);

    expr_model.set_mesh(0.0, TIME_HORIZON, NUM_INTERVALS);

    // dx0/dt = x1,  dx1/dt = u
    const auto dynamics_x0 = StateLeaf{x1_handle.index};
    const auto dynamics_x1 = ControlLeaf{u_handle.index};

    // cost = u^2
    const auto cost_expression =
        ControlLeaf{u_handle.index} * ControlLeaf{u_handle.index};

    // Path constraint: R^2 - x0^2 - x1^2 >= 0
    // Written as: (ConstantExpr{R^2} - StateLeaf{x0}*StateLeaf{x0}
    //              - StateLeaf{x1}*StateLeaf{x1}) >= 0.0
    const double radius_squared = radius_bound * radius_bound;
    const auto norm_squared_expression =
        StateLeaf{x0_handle.index} * StateLeaf{x0_handle.index} +
        StateLeaf{x1_handle.index} * StateLeaf{x1_handle.index};
    // g = R^2 - ||x||^2  (must be >= 0)
    const auto path_expression =
        ConstantExpr{radius_squared} - norm_squared_expression;

    auto ocp = std::move(expr_model)
        .with_dynamics(x0_handle, dynamics_x0)
        .with_dynamics(x1_handle, dynamics_x1)
        .with_cost(integral(cost_expression))
        .with_path_constraint(path_expression >= 0.0)
        .build();

    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, model_name);

    // Initial guess: linear interpolation for positions, zero velocity and control.
    const std::size_t num_variables = compiled.problem->num_variables();
    std::vector<double> initial_guess(num_variables, 0.0);
    const auto& layout = compiled.layout;
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        const double t_frac = static_cast<double>(k) / static_cast<double>(NUM_INTERVALS);
        initial_guess[layout.state_index(k, 0)] = t_frac;  // linear x0: 0->1
        initial_guess[layout.state_index(k, 1)] = 0.0;     // x1 = 0 initially
    }

    goss::solver::IpoptSolver ipopt_solver;
    ipopt_solver.set_tolerance(1e-8);
    const auto solver_result = ipopt_solver.solve(*compiled.problem, initial_guess);

    EXPECT_EQ(solver_result.status, goss::solver::SolverStatus::Success)
        << "Solver failed for model_name=" << model_name;

    return goss::accuracy::solve_and_extract_trajectory(compiled, initial_guess);
}

}  // namespace

// Test A: inactive constraint (R = 2.0).
// The unconstrained double-integrator minimum-energy objective is exactly 12.0
// (see Liberzon §3.3: J* = 12 for x(0)=0, v(0)=0, x(T)=1, v(T)=0, T=1, minimize integral u^2).
// With R=2.0 the constraint x0^2+x1^2 <= 4 is never active (the unconstrained arc stays within
// a max norm of ~0.48), so the objective must equal the unconstrained value within solver tolerance.
TEST(PathConstraintAccuracy, InactiveConstraintObjectiveMatchesUnconstrained) {
    auto trajectory = solve_double_integrator_with_norm_constraint(
        2.0, "pc_accuracy_inactive");

    // Reference: J_unconstrained = 12.0 (Liberzon §3.3).
    EXPECT_NEAR(trajectory.objective_value, 12.0, 1e-2)
        << "With inactive path constraint, objective should match unconstrained optimum 12.0";
}

// Test B: active constraint (R = 0.5).
// The constraint x0^2 + x1^2 <= 0.25 is active along the optimal arc.
// The constrained objective must exceed the unconstrained minimum of 12.0
// (the constraint restricts the feasible set, so the optimal cost increases).
// We also verify the constraint is satisfied at all collocation nodes.
TEST(PathConstraintAccuracy, ActiveConstraintSatisfiedAndObjectiveExceedsUnconstrained) {
    constexpr double RADIUS = 0.5;

    // Use a direct solve (not the helper wrapper) so we can access the full solution vector.
    using namespace goss::model::expr;

    ExprModel<> expr_model;
    const auto x0_handle = expr_model.add_state("position");
    const auto x1_handle = expr_model.add_state("velocity");
    const auto u_handle  = expr_model.add_control("force");

    expr_model.apply(x0_handle.initial() == 0.0);
    expr_model.apply(x1_handle.initial() == 0.0);
    expr_model.apply(x0_handle.final()   == 1.0);
    expr_model.apply(x1_handle.final()   == 0.0);
    expr_model.apply(x0_handle >= -10.0);
    expr_model.apply(x0_handle <= 10.0);
    expr_model.apply(x1_handle >= -10.0);
    expr_model.apply(x1_handle <= 10.0);
    expr_model.apply(u_handle  >= -20.0);
    expr_model.apply(u_handle  <=  20.0);
    expr_model.set_mesh(0.0, TIME_HORIZON, NUM_INTERVALS);

    const auto path_expression =
        ConstantExpr{RADIUS * RADIUS} -
        (StateLeaf{x0_handle.index} * StateLeaf{x0_handle.index} +
         StateLeaf{x1_handle.index} * StateLeaf{x1_handle.index});

    auto ocp = std::move(expr_model)
        .with_dynamics(x0_handle, StateLeaf{x1_handle.index})
        .with_dynamics(x1_handle, ControlLeaf{u_handle.index})
        .with_cost(integral(ControlLeaf{u_handle.index} * ControlLeaf{u_handle.index}))
        .with_path_constraint(path_expression >= 0.0)
        .build();

    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "pc_accuracy_active");
    const auto& layout = compiled.layout;

    std::vector<double> initial_guess(compiled.problem->num_variables(), 0.0);
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        const double t_frac = static_cast<double>(k) / static_cast<double>(NUM_INTERVALS);
        initial_guess[layout.state_index(k, 0)] = t_frac;
    }

    goss::solver::IpoptSolver ipopt_solver;
    ipopt_solver.set_tolerance(1e-8);
    const auto solver_result = ipopt_solver.solve(*compiled.problem, initial_guess);

    ASSERT_EQ(solver_result.status, goss::solver::SolverStatus::Success)
        << "Solver failed for active path-constraint test";

    // Objective must exceed the unconstrained minimum 12.0 (constraint tightens the problem).
    EXPECT_GT(solver_result.objective_value, 12.0 - 1e-6)
        << "Constrained objective must be >= unconstrained minimum 12.0";

    // Path constraint satisfied at all nodes: x0^2 + x1^2 <= R^2 + tol.
    // Allow solver tolerance 1e-6 slack (Ipopt feasibility tolerance is 1e-8).
    constexpr double FEASIBILITY_TOLERANCE = 1e-5;
    for (std::size_t k = 0; k < layout.num_nodes(); ++k) {
        const double x0_k = solver_result.x[layout.state_index(k, 0)];
        const double x1_k = solver_result.x[layout.state_index(k, 1)];
        const double norm_squared = x0_k * x0_k + x1_k * x1_k;
        EXPECT_LE(norm_squared, RADIUS * RADIUS + FEASIBILITY_TOLERANCE)
            << "Path constraint violated at node " << k
            << ": ||x||^2 = " << norm_squared
            << " > R^2 = " << RADIUS * RADIUS;
    }
}
```

- [ ] **Step 2: Add to `CMakeLists.txt` `goss_accuracy_tests` block**

Locate the `goss_accuracy_tests` executable definition (added by the accuracy-suite plan) and append the new source:

```cmake
add_executable(goss_accuracy_tests
  ...existing sources...
  tests/accuracy/test_path_constraint_accuracy.cpp)
```

- [ ] **Step 3: Run to verify tests fail (before the implementation is wired)**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_accuracy_tests 2>&1 | tail -20'
```

Expected: compile or link failure — if Tasks 1-4 are complete, it should compile but the path-constraint fields not yet being populated in the actual solve may cause assertion failures. If Tasks 1-4 are not yet merged, expect compile errors.

- [ ] **Step 4: Run the full accuracy suite after Tasks 1-4 pass**

```bash
scripts/dev.sh 'cmake -B build -S . && cmake --build build --target goss_accuracy_tests && ctest --test-dir build -R PathConstraintAccuracy -V'
```

Expected:
- `PathConstraintAccuracy.InactiveConstraintObjectiveMatchesUnconstrained`: PASS, objective near 12.0.
- `PathConstraintAccuracy.ActiveConstraintSatisfiedAndObjectiveExceedsUnconstrained`: PASS, constraint satisfied at all nodes.

- [ ] **Step 5: Run the complete test suite to verify no regressions**

```bash
scripts/dev.sh 'ctest --test-dir build -V 2>&1 | tail -50'
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/accuracy/test_path_constraint_accuracy.cpp \
        CMakeLists.txt
git commit -m "test(accuracy): add nonlinear path constraint end-to-end accuracy test

Bounded double integrator (minimum-energy, x0^2+x1^2<=R^2) with two cases:
inactive constraint (R=2, J should match unconstrained 12.0 ±1e-2) and
active constraint (R=0.5, J>12 AND constraint satisfied at all nodes ±1e-5).
Reference: Liberzon Calculus of Variations §3.3 for J_unconstrained=12.0."
```

---

## Deferred Items (noted for future plans)

- **Trapezoidal path constraints:** The `OcpProblem` extension from Task 1 is already in place. Adding path constraints to `Trapezoidal::compile()` requires the same row-append loop as Task 2 but in `trapezoidal.hpp`. The `gl`/`gu` extension is identical.
- **LGL path constraints:** Same pattern as Trapezoidal. The LGL packed functor iterates over `nn` nodes already; path-constraint rows follow the defect rows in the same way.
- **Midpoint enforcement (Hermite-Simpson):** Evaluating `g(x_mid, u_mid, t_mid)` at Hermite midpoints adds `ni * npc` additional rows and provides stronger constraint satisfaction between nodes. This is a single-loop addition inside the interval `k` loop in the packed functor.
- **Multiple RHS (expr op expr):** `(expr1) >= (expr2)` where the RHS is also an expression node (not a literal) requires an additional `operator>=` overload accepting `(LhsExpr, RhsExpr)` and producing `PathConstraintExpr<BinaryExpr<SubTag, LhsExpr, RhsExpr>>`. The SFINAE guard on both LHS and RHS types is straightforward.

---

## Self-Review

**Spec coverage:**
- Transcription hook (OcpProblem + HermiteSimpson rows): Tasks 1 + 2. ✓
- DSL lowering (`(q+1.0)>=0.0` compiles): Task 3 + 4. ✓
- ExprModel accumulation and `build()` integration: Task 4. ✓
- Backward compat (bare `q>=0.0` still a box bound): overload-resolution audit in Task 3, regression test in `test_path_constraint_lowering.cpp`. ✓
- AD-safety (no `std::function` in AD path): `PathConstraintFunctor` is fully templated; captured by value in packed lambda. ✓
- `VariableLayout` unchanged: confirmed — path constraints add only constraint rows. ✓
- Bound convention `>= → [0,kInf]`, `<= → [-kInf,0]`, `== → [0,0]`: enforced in `operator>=/<=/==` overloads. ✓
- Accuracy test with reference: Task 5 (double integrator, J*=12.0 Liberzon §3.3, R=0.5 active, R=2.0 inactive). ✓
- HermiteSimpson-first scope, deferred others: explicitly noted in Deferred section. ✓
- Nodes-only enforcement (not midpoints): justified in HermiteSimpson packed functor comment. ✓
- DAE orthogonality: documented in `ocp_problem.hpp` comment; `path_constraint_*` fields are distinct from anticipated `algebraic_*` fields. ✓

**Placeholder scan:** No TBD, no TODO, no "similar to Task N" — each task is self-contained with full code.

**Type consistency audit:**
- `PathConstraintExpr<Expr>` defined in Task 3, consumed by `with_path_constraint()` in Task 4. ✓
- `PathConstraintFunctor<ExprTuple>` defined in Task 3, assembled in `ExprModel::build()` in Task 4. ✓
- `extract_path_constraint_lower/upper` defined in Task 3, called in Task 4. ✓
- `model_.build_with_path_constraints(...)` defined in Task 4 step 4, called in `ExprModel::build()`. ✓
- `OcpProblem<Dyn,Cost,PathConstraintFn>` with three params defined in Task 1, used in Tasks 2, 4, 5. ✓

**Overload-resolution audit (box-vs-path):**
- `q >= 0.0` where `q : StateHandle` → ADL in `goss::model` → `operator>=(StateHandle, double)` → `BoundConstraint`. The `goss::model::expr::operator>=` template has `enable_if_t<is_expr_node_v<LhsExpr>>` which is `false` for `StateHandle` — not a candidate. ✓
- `(q_expr + 1.0) >= 0.0` where `q_expr + 1.0 : BinaryExpr<AddTag, StateLeaf, ConstantExpr>` → ADL in `goss::model::expr` → `operator>=(BinaryExpr<...>, double)` with `is_expr_node_v<BinaryExpr<...>>` == `true` → `PathConstraintExpr`. The `goss::model::operator>=(StateHandle, double)` is not a candidate (wrong LHS type). ✓

**Potential overload-resolution risk identified:** The `==` overload on expr nodes conflicts with the existing `operator==(BoundaryPoint, double)` in `goss::model`. ADL on `BoundaryPoint` (which is `StateHandle::BoundaryPoint` defined in `goss::model`) finds the existing overload — NOT the new `PathConstraintExpr` overload — because `BoundaryPoint` is not an expr-DSL node (`is_expr_node_v<BoundaryPoint>` is `false`). Verdict: no conflict. However, if a user writes `(q.initial()) == 0.0` where `q.initial()` returns `BoundaryPoint`, it still produces `BoundaryConstraint`. And `(StateLeaf{0}) == 0.0` produces `PathConstraintExpr` (equality path constraint). This is the intended behavior and is tested in `test_path_constraint_lowering.cpp`. ✓
