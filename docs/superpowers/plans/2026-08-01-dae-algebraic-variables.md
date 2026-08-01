# DAE Algebraic-Variable Derived Quantities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce Flavor 2 algebraic-variable (DAE-style) derived quantities to the goss framework: algebraic variables are NLP decision variables solved implicitly via per-node residual constraints `g(x, u, z_alg, t) = 0` rather than by explicit inline expressions.

**Architecture:** Four coordinated changes build on each other. (1) `OcpProblem` gains optional algebraic metadata (new optional fields defaulting to empty/zero — no template parameter change, no existing user breakage). (2) `VariableLayout` gains a `num_algebraic` dimension and `algebraic_index(node, j)` accessor, extending the per-node stride from `ns + nc` to `ns + nc + na`; existing `state_index` and `control_index` formulas are algebraically unchanged because algebraic slots are appended after controls within each node group. (3) `HermiteSimpson::compile` is extended to pack algebraic residual constraints after the defect constraints (Trapezoidal and LGL are deferred with a documented note). (4) `ComposedModel` gains `add_algebraic` on `Component` and a new `build` overload that wires algebraic metadata through to `OcpProblem`. AD safety is preserved throughout: residual functors are concrete generic-lambda template parameters, never `std::function`.

**Tech Stack:** C++17, header-only goss, CppADCG JIT via `goss::ad::CppADCGBackend`, GoogleTest v1.14.0, IPOPT via `goss::solver::IpoptSolver`, `scripts/dev.sh` for all build/test commands.

## Global Constraints

- C++17 (`CMAKE_CXX_STANDARD 17`, `CMAKE_CXX_STANDARD_REQUIRED ON`) — no C++20 features.
- Container-first — ALL cmake/ctest invocations via `scripts/dev.sh '<command>'`; never run cmake/ctest directly on the host.
- Header-only production code under `include/goss/` — no new `.cpp` files needed for the framework itself; test `.cpp` files go under `tests/`.
- GoogleTest — all tests use `TEST(Suite, CaseName)` + `ASSERT_*/EXPECT_*`; no custom test runner.
- Verbose, descriptive names — `num_algebraic_variables` not `na` as member names; `algebraic_lower_bounds` not `zalb`; `algebraic_residuals_functor` not `gfn`.
- AD-safety invariant (LOAD-BEARING): algebraic residual functors are concrete generic-lambda template parameters in the AD path. `std::function` is used ONLY for the double-typed validation lambda stored in `AlgebraicEntry::validation_fn` inside `Component`. The generic template lambda passed to `build()` overloads is captured by value into the packed functor and must never be type-erased via `std::function` before the CppADCG recording step.
- Backward-compatibility: existing `OcpProblem` aggregate initialization, `VariableLayout` constructors, and all three scheme `compile()` overloads must remain unchanged for callers with `num_algebraic == 0`. Every existing test must continue to compile and pass without modification.
- `ComponentError` for API misuse (duplicate algebraic name, mismatched bound vectors, etc.).
- Each `Scheme::compile()` call requires a unique `model_name` string (CppADCG uses it as a shared-library filename; collisions cause silent errors).
- Pre-condition: the accuracy validation suite plan (`docs/superpowers/plans/2026-08-01-accuracy-validation-suite.md`, target `goss_accuracy_tests`, helpers in `tests/accuracy/accuracy_helpers.hpp`) MUST be merged and all `goss_accuracy_tests` passing before Task 4 (the end-to-end DAE accuracy test) is executed.
- Scope decision: **HermiteSimpson gets algebraic support in this plan. Trapezoidal and LegendreGaussLobatto are explicitly deferred.** Justification: HermiteSimpson is the primary scheme used by the composition layer (all composition solve tests use it); extending all three simultaneously would triple the residual-injection surface area and make the first PR harder to review. Trapezoidal and LGL extensions follow naturally once the pattern is established. A `static_assert` / `TranscriptionError` guard is NOT added to the deferred schemes — they simply ignore `num_algebraic` in `OcpProblem` (zero by default), so existing users are unaffected.

---

## Scope: What Is and Is NOT Built Here

**In scope (this plan):**
- `AlgebraicEntry` struct and `Component::add_algebraic(name, validation_fn, lower, upper)` API.
- `AlgebraicHandle` opaque handle.
- `OcpProblem` optional algebraic fields: `num_algebraic`, `algebraic_residuals_functor` (template param `AlgResFn`), `algebraic_lower_bounds`, `algebraic_upper_bounds`.
- `VariableLayout` extended constructor and `algebraic_index(node, j)` accessor.
- `HermiteSimpson::compile` algebraic residual injection.
- `ComposedModel::build` new overload with algebraic residual generic lambda.
- `Component::add_algebraic` validation (name uniqueness, bound checks).
- `ComposedModel::validate_algebraic_dimensions()` using double validation lambdas.
- End-to-end DAE solve test with known reference values (see Task 4).

**Explicitly deferred (future plans):**
- Trapezoidal algebraic support.
- LegendreGaussLobatto algebraic support.
- Multi-component algebraic variables.
- Algebraic variables that depend on other derived quantities (topo ordering for algebraics).
- Initial-guess seeding for algebraic variables beyond the flat-value approach described below.

---

## File Structure

| File | Created / Modified | Responsibility |
|---|---|---|
| `include/goss/model/component.hpp` | Modified | Add `AlgebraicHandle`, `AlgebraicEntry`, `Component::add_algebraic()`, `Component::algebraic_entries()`, `Component::num_algebraic()`, `Component::evaluate_algebraic_residuals()`. |
| `include/goss/transcription/ocp_problem.hpp` | Modified | Add three new optional fields to `OcpProblem`: `num_algebraic` (default 0), `algebraic_lower_bounds` (default empty), `algebraic_upper_bounds` (default empty). Add a third template parameter `AlgResFn` with a no-op default type. |
| `include/goss/transcription/variable_layout.hpp` | Modified | Add `num_algebraic_` member, extend constructor, add `algebraic_index(node, j)` accessor, update `variables_per_node()` and `total_variables()`. |
| `include/goss/transcription/hermite_simpson.hpp` | Modified | Extend `compile()` to extract algebraic variables from `z`, call `ocp.algebraic_residuals_functor`, append residual rows to `outputs`, extend `zl`/`zu`/`gl`/`gu`. |
| `include/goss/model/composed_model.hpp` | Modified | Add `ComposedModel::build` overload accepting an algebraic residual generic lambda. Add `validate_algebraic_dimensions()`. Extend `build_internal_model` to populate algebraic fields. |
| `tests/transcription/test_variable_layout.cpp` | Modified | Add tests for `algebraic_index`, updated stride, zero-algebraic backward compat. |
| `tests/transcription/test_hermite_simpson.cpp` | Modified | Add DAE residual constraint test: verify `num_constraints` increases correctly, constraint bounds are `[0,0]`. |
| `tests/model/test_component.cpp` | Modified | Add unit tests for `add_algebraic`, `AlgebraicEntry` storage, validation lambda evaluation, `ComponentError` on duplicate name. |
| `tests/model/test_composed_model.cpp` | Modified | Add tests for `ComposedModel` algebraic metadata collection, `validate_algebraic_dimensions`. |
| `tests/accuracy/test_dae_accuracy.cpp` | Created | End-to-end DAE solve test with analytic reference. Requires accuracy suite to be merged first. |
| `CMakeLists.txt` | Modified | Add `tests/accuracy/test_dae_accuracy.cpp` to `goss_accuracy_tests` sources. |

---

## OcpProblem Extension Design

**Decision: new optional fields + new template parameter with no-op default. No breaking change.**

The cleanest backward-compatible approach adds a third template parameter `AlgResFn` with a default type `NoAlgebraicResiduals` (a no-op struct). Callers that do not use algebraic variables never name this parameter — deduction still works. The three new fields are:

```cpp
// In include/goss/transcription/ocp_problem.hpp

namespace goss::transcription {

/// No-op algebraic residuals functor for problems with num_algebraic == 0.
/// Used as the default AlgResFn template argument so existing OcpProblem<Dyn,Cost>
/// users are unaffected.
struct NoAlgebraicResiduals {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& /*x*/,
                               const std::vector<T>& /*u*/,
                               const std::vector<T>& /*alg_vars*/,
                               T /*t*/) const {
        return {};
    }
};

template <typename DynamicsFn, typename CostFn,
          typename AlgResFn = NoAlgebraicResiduals>
struct OcpProblem {
    std::size_t num_states;
    std::size_t num_controls;
    DynamicsFn dynamics;
    CostFn cost;
    Mesh mesh;
    std::vector<double> state_lower;
    std::vector<double> state_upper;
    std::vector<double> control_lower;
    std::vector<double> control_upper;
    std::vector<double> initial_state;
    std::vector<double> initial_state_fixed;
    std::vector<double> final_state;
    std::vector<double> final_state_fixed;
    // ---- Algebraic-variable (DAE) fields — default to "no algebraics" ----
    // num_algebraic == 0 means this is a standard ODE problem; all three schemes
    // remain unaffected by these fields when num_algebraic == 0.
    std::size_t num_algebraic = 0;
    AlgResFn algebraic_residuals_functor = AlgResFn{};
    // Per-algebraic-variable bounds, size == num_algebraic.
    std::vector<double> algebraic_lower_bounds;   // default empty
    std::vector<double> algebraic_upper_bounds;   // default empty
};

}  // namespace goss::transcription
```

Existing aggregate-initialization callsites (`OcpProblem<Dyn,Cost>{...}` with 12 positional members) remain valid because the new fields have default values and C++17 aggregate initialization allows trailing defaults to be omitted. Existing template deduction (`auto ocp = model.build(dynamics, cost)`) still works because `Model::build` returns `OcpProblem<DynamicsFn, CostFn>` (two-param deduction, `AlgResFn` defaults to `NoAlgebraicResiduals`).

---

## VariableLayout Stride Change

**New index formulas (with `na = num_algebraic_`):**

```
variables_per_node() = ns + nc + na

state_index(node, i)     = node * (ns + nc + na) + i
control_index(node, j)   = node * (ns + nc + na) + ns + j
algebraic_index(node, k) = node * (ns + nc + na) + ns + nc + k
total_variables()        = num_nodes_ * (ns + nc + na)
```

When `na == 0`, `variables_per_node()` is identical to the current `ns + nc`, and `state_index` / `control_index` produce exactly the same values as today. **This is the key backward-compatibility guarantee.** No existing test exercises `algebraic_index` yet, and no existing formula needs to change for `na == 0`.

---

## Task 1: `AlgebraicEntry`, `AlgebraicHandle`, and `Component::add_algebraic`

**Files:**
- Modify: `include/goss/model/component.hpp`
- Modify: `tests/model/test_component.cpp`

**Interfaces:**
- Consumes: `ComponentError` (already in `goss/model/errors.hpp`), `transcription::kInf`
- Produces:
  - `struct AlgebraicHandle { std::size_t index; constexpr operator std::size_t() const noexcept; }`
  - `struct AlgebraicEntry { std::string name; std::function<double(const std::vector<double>& x, const std::vector<double>& u, const std::vector<double>& alg_vars, double t)> validation_fn; double lower_bound; double upper_bound; }`
  - `AlgebraicHandle Component::add_algebraic(const std::string& name, std::function<double(const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double)> validation_fn, double lower, double upper)`
  - `std::size_t Component::num_algebraic() const`
  - `const std::vector<AlgebraicEntry>& Component::algebraic_entries() const`
  - `double Component::evaluate_algebraic_residual(std::size_t j, const std::vector<double>& x, const std::vector<double>& u, const std::vector<double>& alg_vars, double t) const`

- [ ] **Step 1: Write failing unit tests for `add_algebraic`**

Add to `tests/model/test_component.cpp`:

```cpp
#include "goss/model/component.hpp"
#include "goss/model/errors.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <functional>

TEST(ComponentAlgebraic, AddAlgebraicStoresEntry) {
    goss::model::Component component("test_comp");
    // Validation lambda: the algebraic residual g(x, u, z_alg, t) = z_alg[0] - x[0]*x[0]
    // Enforces z_alg[0] = x[0]^2 when the solver drives g to zero.
    auto residual_validation_fn = [](
            const std::vector<double>& x,
            const std::vector<double>& /*u*/,
            const std::vector<double>& alg_vars,
            double /*t*/) -> double {
        return alg_vars[0] - x[0] * x[0];
    };
    auto handle = component.add_algebraic("x_squared", residual_validation_fn, 0.0, 1e19);
    EXPECT_EQ(component.num_algebraic(), 1u);
    EXPECT_EQ(static_cast<std::size_t>(handle), 0u);
    const auto& entries = component.algebraic_entries();
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].name, "x_squared");
    EXPECT_DOUBLE_EQ(entries[0].lower_bound, 0.0);
    EXPECT_DOUBLE_EQ(entries[0].upper_bound, 1e19);
}

TEST(ComponentAlgebraic, DuplicateAlgebraicNameThrows) {
    goss::model::Component component("test_comp");
    auto first_fn = [](const std::vector<double>&, const std::vector<double>&,
                       const std::vector<double>&, double) -> double { return 0.0; };
    auto second_fn = [](const std::vector<double>&, const std::vector<double>&,
                        const std::vector<double>&, double) -> double { return 0.0; };
    component.add_algebraic("my_alg", first_fn, -1e19, 1e19);
    EXPECT_THROW(component.add_algebraic("my_alg", second_fn, -1e19, 1e19),
                 goss::model::ComponentError);
}

TEST(ComponentAlgebraic, DuplicateNameWithStateThrows) {
    goss::model::Component component("test_comp");
    component.add_state("position");
    auto residual_fn = [](const std::vector<double>&, const std::vector<double>&,
                          const std::vector<double>&, double) -> double { return 0.0; };
    // "position" is already registered as a state — should throw.
    EXPECT_THROW(component.add_algebraic("position", residual_fn, -1e19, 1e19),
                 goss::model::ComponentError);
}

TEST(ComponentAlgebraic, EvaluateAlgebraicResidualInvokesValidationLambda) {
    goss::model::Component component("test_comp");
    // g(x, u, z_alg, t) = z_alg[0] - 2.0 * x[0]
    auto residual_fn = [](const std::vector<double>& x,
                          const std::vector<double>& /*u*/,
                          const std::vector<double>& alg_vars,
                          double /*t*/) -> double {
        return alg_vars[0] - 2.0 * x[0];
    };
    component.add_algebraic("twice_x", residual_fn, -1e19, 1e19);
    // At x[0]=3.0, alg_vars[0]=6.0: residual = 6.0 - 2*3.0 = 0.0 (satisfied)
    std::vector<double> x{3.0};
    std::vector<double> u{};
    std::vector<double> alg_vars{6.0};
    double residual = component.evaluate_algebraic_residual(0, x, u, alg_vars, 0.0);
    EXPECT_DOUBLE_EQ(residual, 0.0);
    // At alg_vars[0]=5.0: residual = 5.0 - 6.0 = -1.0 (not satisfied)
    alg_vars[0] = 5.0;
    residual = component.evaluate_algebraic_residual(0, x, u, alg_vars, 0.0);
    EXPECT_DOUBLE_EQ(residual, -1.0);
}

TEST(ComponentAlgebraic, NumAlgebraicIsZeroByDefault) {
    goss::model::Component component("test_comp");
    component.add_state("x");
    EXPECT_EQ(component.num_algebraic(), 0u);
}
```

- [ ] **Step 2: Run to verify tests fail**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests 2>&1 | tail -30'
```

Expected: compile error — `'add_algebraic'` not a member of `Component`, `num_algebraic` not found.

- [ ] **Step 3: Implement `AlgebraicHandle`, `AlgebraicEntry`, and `Component::add_algebraic`**

In `include/goss/model/component.hpp`, add directly after the closing brace of the `DerivedHandle` struct definition (after line 23):

```cpp
/// Opaque handle to an algebraic-variable (DAE Flavor 2) entry, analogous to DerivedHandle.
struct AlgebraicHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }
};

/// Metadata for one algebraic variable registered via Component::add_algebraic().
/// The validation_fn stores the double-typed residual g(x, u, z_alg, t) for
/// validation-path evaluation only. The AD path uses a generic template lambda
/// passed as a concrete template parameter to ComposedModel::build() — never
/// type-erased here.
struct AlgebraicEntry {
    std::string name;
    /// Residual function: g(x, u, alg_vars, t) → double.
    /// The solver enforces g == 0 at every collocation node.
    std::function<double(
        const std::vector<double>& x,
        const std::vector<double>& u,
        const std::vector<double>& alg_vars,
        double t)> validation_fn;
    double lower_bound;
    double upper_bound;
};
```

Then, in the `Component` class public section, add after the `add_derived` method (after line 130):

```cpp
    /// Register an algebraic variable (DAE Flavor 2).
    ///
    /// The validation_fn (double-typed) is stored for validation-path evaluation only.
    /// The actual AD-safe residual is a concrete generic lambda passed to
    /// ComposedModel::build() — see the AD-safety invariant in the plan.
    ///
    /// The solver enforces validation_fn(x, u, alg_vars, t) == 0 at every
    /// collocation node by adding one equality constraint per algebraic variable per node.
    ///
    /// lower and upper are bounds on the algebraic variable's value (not on the residual).
    AlgebraicHandle add_algebraic(
            const std::string& name,
            std::function<double(const std::vector<double>&,
                                 const std::vector<double>&,
                                 const std::vector<double>&,
                                 double)> validation_fn,
            double lower,
            double upper) {
        ensure_name_unique_in_component(name);
        if (lower > upper) {
            throw ComponentError(
                "Component '" + component_name_ + "': add_algebraic(\"" + name +
                "\"): lower bound " + std::to_string(lower) +
                " > upper bound " + std::to_string(upper));
        }
        const std::size_t local_index = algebraic_entries_.size();
        algebraic_entries_.push_back(AlgebraicEntry{
            name,
            std::move(validation_fn),
            lower,
            upper
        });
        return AlgebraicHandle{local_index};
    }
```

Add accessors after `num_derived()` (after line 212):

```cpp
    std::size_t num_algebraic() const { return algebraic_entries_.size(); }

    const std::vector<AlgebraicEntry>& algebraic_entries() const { return algebraic_entries_; }

    /// Invoke the double-path validation residual for algebraic variable j.
    /// For validation use only; the AD path uses the generic lambda passed to build().
    double evaluate_algebraic_residual(
            std::size_t j,
            const std::vector<double>& global_x,
            const std::vector<double>& global_u,
            const std::vector<double>& alg_vars,
            double t) const {
        if (j >= algebraic_entries_.size()) {
            throw ComponentError(
                "Component '" + component_name_ +
                "': evaluate_algebraic_residual: index " + std::to_string(j) +
                " out of range (" + std::to_string(algebraic_entries_.size()) +
                " algebraic entries)");
        }
        return algebraic_entries_[j].validation_fn(global_x, global_u, alg_vars, t);
    }
```

Extend `ensure_name_unique_in_component` to also check `algebraic_entries_`:

```cpp
    void ensure_name_unique_in_component(const std::string& name) const {
        for (const auto& entry : owned_states_) {
            if (entry.name == name) {
                throw ComponentError(
                    "Component '" + component_name_ + "': duplicate name '" + name + "'");
            }
        }
        for (const auto& entry : derived_entries_) {
            if (entry.name == name) {
                throw ComponentError(
                    "Component '" + component_name_ + "': duplicate name '" + name + "'");
            }
        }
        for (const auto& entry : algebraic_entries_) {
            if (entry.name == name) {
                throw ComponentError(
                    "Component '" + component_name_ + "': duplicate name '" + name + "'");
            }
        }
    }
```

Add the private data member after `has_cost_`:

```cpp
    std::vector<AlgebraicEntry> algebraic_entries_;
```

- [ ] **Step 4: Run to verify tests pass**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests && ctest --test-dir build --output-on-failure -R "ComponentAlgebraic" 2>&1 | tail -30'
```

Expected: all `ComponentAlgebraic.*` tests pass. All pre-existing model tests pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add include/goss/model/component.hpp tests/model/test_component.cpp
git commit -m "feat(model): add AlgebraicEntry/AlgebraicHandle and Component::add_algebraic for DAE Flavor 2"
```

---

## Task 2: `OcpProblem` Algebraic Fields and `VariableLayout` Extension

**Files:**
- Modify: `include/goss/transcription/ocp_problem.hpp`
- Modify: `include/goss/transcription/variable_layout.hpp`
- Modify: `tests/transcription/test_variable_layout.cpp`

**Interfaces:**
- Consumes: `TranscriptionError`, existing `VariableLayout(num_states, num_controls, num_nodes)` constructor signature
- Produces:
  - `struct NoAlgebraicResiduals` with `template<typename T> std::vector<T> operator()(...)` returning empty
  - `OcpProblem<DynamicsFn, CostFn, AlgResFn = NoAlgebraicResiduals>` with new fields `num_algebraic`, `algebraic_residuals_functor`, `algebraic_lower_bounds`, `algebraic_upper_bounds`
  - `VariableLayout(std::size_t num_states, std::size_t num_controls, std::size_t num_nodes, std::size_t num_algebraic = 0)` — extended constructor, zero default preserves backward compat
  - `std::size_t VariableLayout::num_algebraic() const`
  - `std::size_t VariableLayout::algebraic_index(std::size_t node, std::size_t alg_var) const`
  - Updated `variables_per_node()` = `ns + nc + na`
  - Updated `total_variables()` = `num_nodes_ * (ns + nc + na)`

- [ ] **Step 1: Write failing tests**

Add to `tests/transcription/test_variable_layout.cpp`:

```cpp
TEST(VariableLayout, AlgebraicIndexAppendsAfterControls) {
    // 2 states, 1 control, 1 algebraic, 3 nodes.
    // Per-node stride = 2+1+1 = 4.
    // node 0: x0=0, x1=1, u0=2, alg0=3
    // node 1: x0=4, x1=5, u0=6, alg0=7
    // node 2: x0=8, x1=9, u0=10, alg0=11
    goss::transcription::VariableLayout layout(2, 1, 3, /*num_algebraic=*/1);
    EXPECT_EQ(layout.total_variables(), 12u);
    EXPECT_EQ(layout.variables_per_node(), 4u);
    EXPECT_EQ(layout.state_index(0, 0), 0u);
    EXPECT_EQ(layout.state_index(0, 1), 1u);
    EXPECT_EQ(layout.control_index(0, 0), 2u);
    EXPECT_EQ(layout.algebraic_index(0, 0), 3u);
    EXPECT_EQ(layout.state_index(1, 0), 4u);
    EXPECT_EQ(layout.control_index(1, 0), 6u);
    EXPECT_EQ(layout.algebraic_index(1, 0), 7u);
    EXPECT_EQ(layout.algebraic_index(2, 0), 11u);
}

TEST(VariableLayout, ZeroAlgebraicPreservesExistingIndices) {
    // With num_algebraic=0 (explicit), indices must be identical to the 3-arg constructor.
    goss::transcription::VariableLayout layout_old(2, 1, 3);
    goss::transcription::VariableLayout layout_new(2, 1, 3, 0);
    EXPECT_EQ(layout_old.total_variables(), layout_new.total_variables());
    EXPECT_EQ(layout_old.state_index(1, 0), layout_new.state_index(1, 0));
    EXPECT_EQ(layout_old.control_index(2, 0), layout_new.control_index(2, 0));
}

TEST(VariableLayout, AlgebraicIndexOutOfRangeThrows) {
    goss::transcription::VariableLayout layout(2, 1, 3, 1);
    EXPECT_THROW(layout.algebraic_index(3, 0), goss::transcription::TranscriptionError);  // node OOB
    EXPECT_THROW(layout.algebraic_index(0, 1), goss::transcription::TranscriptionError);  // alg_var OOB
}

TEST(VariableLayout, AlgebraicIndexWithZeroAlgebraicThrows) {
    // Calling algebraic_index when num_algebraic==0 is always a programming error.
    goss::transcription::VariableLayout layout(2, 1, 3, 0);
    EXPECT_THROW(layout.algebraic_index(0, 0), goss::transcription::TranscriptionError);
}

TEST(OcpProblem, NewAlgebraicFieldsDefaultToEmpty) {
    // Existing two-param OcpProblem must compile unchanged and have zero algebraics.
    auto ocp = goss::transcription::test::make_exponential_decay(1.0, 1.0, 5);
    EXPECT_EQ(ocp.num_algebraic, 0u);
    EXPECT_TRUE(ocp.algebraic_lower_bounds.empty());
    EXPECT_TRUE(ocp.algebraic_upper_bounds.empty());
}
```

- [ ] **Step 2: Run to verify tests fail**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_transcription_tests 2>&1 | tail -30'
```

Expected: compile errors — `algebraic_index` not a member of `VariableLayout`, `num_algebraic` not a member of `OcpProblem`.

- [ ] **Step 3: Extend `OcpProblem` in `include/goss/transcription/ocp_problem.hpp`**

Replace the existing file content with:

```cpp
// include/goss/transcription/ocp_problem.hpp
#pragma once
#include <cstddef>
#include <vector>
#include "goss/transcription/errors.hpp"

namespace goss::transcription {

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

/// No-op algebraic residuals functor for problems with num_algebraic == 0.
/// Used as the default AlgResFn template argument so all existing
/// OcpProblem<Dyn,Cost> users are unaffected — the third template parameter
/// defaults to this type and is never named by existing callers.
struct NoAlgebraicResiduals {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& /*x*/,
                               const std::vector<T>& /*u*/,
                               const std::vector<T>& /*alg_vars*/,
                               T /*t*/) const {
        return {};
    }
};

/// Optimal-control problem description passed from the model DSL layer to a
/// transcription scheme's compile() function.
///
/// Template parameters:
///   DynamicsFn  — generic functor (x, u, t) → vector<T>, the ODE right-hand side.
///   CostFn      — generic functor (x, u, t) → T, the running cost integrand.
///   AlgResFn    — generic functor (x, u, alg_vars, t) → vector<T>, the algebraic
///                 residual vector (size == num_algebraic). Defaults to NoAlgebraicResiduals
///                 for num_algebraic == 0; existing callers never name this parameter.
template <typename DynamicsFn, typename CostFn,
          typename AlgResFn = NoAlgebraicResiduals>
struct OcpProblem {
    std::size_t num_states;
    std::size_t num_controls;
    DynamicsFn dynamics;    // template<T> vector<T>(const vector<T>& x, const vector<T>& u, T t)
    CostFn cost;            // template<T> T(const vector<T>& x, const vector<T>& u, T t)
    Mesh mesh;
    std::vector<double> state_lower;
    std::vector<double> state_upper;
    std::vector<double> control_lower;
    std::vector<double> control_upper;
    std::vector<double> initial_state;
    std::vector<double> initial_state_fixed;   // nonzero => pin node 0 state i
    std::vector<double> final_state;
    std::vector<double> final_state_fixed;     // nonzero => pin last node state i

    // ---- Algebraic-variable (DAE Flavor 2) fields ----
    // When num_algebraic == 0 (the default), all three schemes behave exactly as before:
    // the algebraic_residuals_functor is never called, and the bound vectors are empty.
    std::size_t num_algebraic = 0;
    // Residual functor: g(x, u, alg_vars, t) → vector<T> of size num_algebraic.
    // The solver enforces g[j] == 0 at every collocation node via equality constraints.
    AlgResFn algebraic_residuals_functor = AlgResFn{};
    // Box bounds on each algebraic variable's value (not on the residual).
    // Size must equal num_algebraic.
    std::vector<double> algebraic_lower_bounds;
    std::vector<double> algebraic_upper_bounds;
};

}  // namespace goss::transcription
```

- [ ] **Step 4: Extend `VariableLayout` in `include/goss/transcription/variable_layout.hpp`**

Replace the file content with:

```cpp
#pragma once
#include <cstddef>
#include <string>
#include "goss/transcription/errors.hpp"

namespace goss::transcription {

/// Maps (node, state|control|algebraic) coordinates to flat decision-vector indices.
///
/// Layout is grouped by node; within each node, states precede controls which precede
/// algebraic variables:
///   z = [ x_0, u_0, alg_0,  x_1, u_1, alg_1,  ...,  x_{Nn-1}, u_{Nn-1}, alg_{Nn-1} ]
///
/// Per-node stride = num_states + num_controls + num_algebraic.
///
/// When num_algebraic == 0 (the default), the layout is identical to the pre-algebraic
/// layout (stride = num_states + num_controls), and state_index / control_index produce
/// exactly the same values. This is the single source of truth for z packing; every
/// scheme and test must use it so indices never diverge.
class VariableLayout {
 public:
    /// Primary constructor. num_algebraic defaults to 0 for backward compatibility.
    VariableLayout(std::size_t num_states, std::size_t num_controls,
                   std::size_t num_nodes, std::size_t num_algebraic = 0)
        : num_states_(num_states), num_controls_(num_controls),
          num_nodes_(num_nodes), num_algebraic_(num_algebraic) {
        if (num_states_ == 0) {
            throw TranscriptionError("VariableLayout: num_states must be >= 1");
        }
        if (num_nodes_ < 2) {
            throw TranscriptionError("VariableLayout: num_nodes must be >= 2 (need >= 1 interval)");
        }
    }

    std::size_t num_states()    const { return num_states_; }
    std::size_t num_controls()  const { return num_controls_; }
    std::size_t num_nodes()     const { return num_nodes_; }
    std::size_t num_algebraic() const { return num_algebraic_; }

    /// Number of decision variables per node: states + controls + algebraic variables.
    std::size_t variables_per_node() const {
        return num_states_ + num_controls_ + num_algebraic_;
    }

    /// Total decision variables across all nodes.
    std::size_t total_variables() const { return num_nodes_ * variables_per_node(); }

    /// Index of state component i at the given node.
    /// Formula: node * (ns + nc + na) + i
    std::size_t state_index(std::size_t node, std::size_t state) const {
        if (node >= num_nodes_)
            throw TranscriptionError("VariableLayout::state_index: node out of range");
        if (state >= num_states_)
            throw TranscriptionError("VariableLayout::state_index: state out of range");
        return node * variables_per_node() + state;
    }

    /// Index of control component j at the given node.
    /// Formula: node * (ns + nc + na) + ns + j
    std::size_t control_index(std::size_t node, std::size_t control) const {
        if (node >= num_nodes_)
            throw TranscriptionError("VariableLayout::control_index: node out of range");
        if (control >= num_controls_)
            throw TranscriptionError("VariableLayout::control_index: control out of range");
        return node * variables_per_node() + num_states_ + control;
    }

    /// Index of algebraic variable k at the given node.
    /// Formula: node * (ns + nc + na) + ns + nc + k
    /// Throws if num_algebraic_ == 0 (programming error: no algebraics registered).
    std::size_t algebraic_index(std::size_t node, std::size_t alg_var) const {
        if (num_algebraic_ == 0)
            throw TranscriptionError(
                "VariableLayout::algebraic_index: no algebraic variables registered "
                "(num_algebraic == 0)");
        if (node >= num_nodes_)
            throw TranscriptionError("VariableLayout::algebraic_index: node out of range");
        if (alg_var >= num_algebraic_)
            throw TranscriptionError("VariableLayout::algebraic_index: alg_var out of range");
        return node * variables_per_node() + num_states_ + num_controls_ + alg_var;
    }

 private:
    std::size_t num_states_;
    std::size_t num_controls_;
    std::size_t num_nodes_;
    /// Number of algebraic variables per node. Zero means this is a standard ODE problem;
    /// VariableLayout behaves exactly as the pre-algebraic version when this is zero.
    std::size_t num_algebraic_;
};

}  // namespace goss::transcription
```

- [ ] **Step 5: Run to verify tests pass**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_transcription_tests && ctest --test-dir build --output-on-failure -R "VariableLayout|OcpProblem" 2>&1 | tail -40'
```

Expected: all `VariableLayout.*` and `OcpProblem.*` tests pass. All `HermiteSimpson.*`, `Trapezoidal.*`, and `LGL.*` tests pass unchanged (they construct `VariableLayout` with 3 args; the default `num_algebraic=0` preserves all indices).

- [ ] **Step 6: Commit**

```bash
git add include/goss/transcription/ocp_problem.hpp include/goss/transcription/variable_layout.hpp tests/transcription/test_variable_layout.cpp
git commit -m "feat(transcription): add algebraic fields to OcpProblem and algebraic_index to VariableLayout for DAE Flavor 2"
```

---

## Task 3: `HermiteSimpson::compile` Algebraic Residual Injection

**Files:**
- Modify: `include/goss/transcription/hermite_simpson.hpp`
- Modify: `tests/transcription/test_hermite_simpson.cpp`

**Interfaces:**
- Consumes: `OcpProblem<DynamicsFn, CostFn, AlgResFn>` (three template params), `VariableLayout::algebraic_index`, `VariableLayout::total_variables` (updated stride), `TranscriptionError`, `kInf`
- Produces: Extended `HermiteSimpson::compile()` that, when `ocp.num_algebraic > 0`:
  - Constructs `VariableLayout(ns, nc, nn, na)` instead of `VariableLayout(ns, nc, nn)`.
  - Appends `na * nn` algebraic residual constraints to `outputs` after the `ni * ns` defect rows.
  - Extends `zl`/`zu` with per-node algebraic variable bounds.
  - Extends `gl`/`gu` with `na * nn` equality entries `[0.0, 0.0]`.
  - Validates `ocp.algebraic_lower_bounds.size() == na && ocp.algebraic_upper_bounds.size() == na`.

**AD-safety note for this task:** The algebraic residual loop inside `packed` must call `ocp.algebraic_residuals_functor(xk, uk, alg_k, tk)` where `alg_k` is a `std::vector<T>` extracted from `z` via `layout.algebraic_index`. This is a generic call on a concrete functor type — no `std::function` in the hot path. `ocp` is captured by value, so `ocp.algebraic_residuals_functor` is a concrete object of type `AlgResFn`.

**Scope note — deferred schemes:** Trapezoidal and LegendreGaussLobatto are NOT modified in this task. Both schemes' `compile()` functions accept the now-three-param `OcpProblem` template (via `AlgResFn = NoAlgebraicResiduals`), but they continue to construct `VariableLayout(ns, nc, nn)` (using the backward-compatible default `num_algebraic = 0`) and do not call `algebraic_residuals_functor`. Passing an `OcpProblem` with `num_algebraic > 0` to Trapezoidal or LGL will silently drop the algebraic constraints — this is intentional for this plan's scope. A future plan will add a guard `if (ocp.num_algebraic > 0) throw TranscriptionError(...)` to the deferred schemes before extending them.

- [ ] **Step 1: Write failing tests for algebraic constraint injection**

Add to `tests/transcription/test_hermite_simpson.cpp`:

```cpp
namespace {

/// Minimal OcpProblem with 1 state, 0 controls, 1 algebraic variable.
/// Dynamics: dx/dt = 0 (trivial, no movement).
/// Algebraic residual: g(x, u, z_alg, t) = z_alg[0] - 2.0*x[0]
/// This enforces z_alg[0] = 2*x[0] at every node.
/// With x(0) = 3.0 fixed and dx/dt = 0, the solution is x(t) = 3.0 everywhere,
/// so z_alg(t) = 6.0 everywhere.
struct TrivialAlgebraicDynamics {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& /*x*/,
                               const std::vector<T>& /*u*/,
                               T /*t*/) const {
        return { T(0.0) };  // dx/dt = 0
    }
};
struct ZeroCostAlg {
    template <typename T>
    T operator()(const std::vector<T>& /*x*/,
                 const std::vector<T>& /*u*/,
                 T /*t*/) const {
        return T(0);
    }
};
struct TwiceXResidual {
    // g(x, u, z_alg, t) = z_alg[0] - 2*x[0]
    // Enforces z_alg[0] = 2*x[0] when driven to zero by the solver.
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x,
                               const std::vector<T>& /*u*/,
                               const std::vector<T>& alg_vars,
                               T /*t*/) const {
        return { alg_vars[0] - T(2.0) * x[0] };
    }
};

goss::transcription::OcpProblem<TrivialAlgebraicDynamics, ZeroCostAlg, TwiceXResidual>
make_twice_x_algebraic_ocp(double x0_val, std::size_t num_intervals) {
    goss::transcription::OcpProblem<TrivialAlgebraicDynamics, ZeroCostAlg, TwiceXResidual> ocp;
    ocp.num_states = 1;
    ocp.num_controls = 0;
    ocp.dynamics = TrivialAlgebraicDynamics{};
    ocp.cost = ZeroCostAlg{};
    ocp.mesh = goss::transcription::Mesh{0.0, 1.0, num_intervals};
    ocp.state_lower = { -1e19 };
    ocp.state_upper = { 1e19 };
    ocp.control_lower = {};
    ocp.control_upper = {};
    ocp.initial_state = { x0_val };
    ocp.initial_state_fixed = { 1.0 };
    ocp.final_state = { 0.0 };
    ocp.final_state_fixed = { 0.0 };
    ocp.num_algebraic = 1;
    ocp.algebraic_residuals_functor = TwiceXResidual{};
    ocp.algebraic_lower_bounds = { -1e19 };
    ocp.algebraic_upper_bounds = { 1e19 };
    return ocp;
}

}  // namespace

TEST(HermiteSimpsonAlgebraic, NumConstraintsIncludesAlgebraicResiduals) {
    // 5 intervals, 6 nodes (nn=6).
    // Defect constraints: ni*ns = 5*1 = 5.
    // Algebraic residual constraints: nn*na = 6*1 = 6.
    // Total constraints = 5 + 6 = 11.
    const std::size_t num_intervals = 5;
    auto ocp = make_twice_x_algebraic_ocp(3.0, num_intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_alg_count");
    const std::size_t nn = num_intervals + 1;
    const std::size_t expected_defects = num_intervals * 1;  // ni * ns
    const std::size_t expected_alg_residuals = nn * 1;       // nn * na
    EXPECT_EQ(compiled.problem->num_constraints(),
              expected_defects + expected_alg_residuals);
}

TEST(HermiteSimpsonAlgebraic, AlgebraicConstraintBoundsAreEqualityZero) {
    const std::size_t num_intervals = 4;
    auto ocp = make_twice_x_algebraic_ocp(3.0, num_intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_alg_bounds");
    const auto& gl = compiled.problem->constraint_lower_bounds();
    const auto& gu = compiled.problem->constraint_upper_bounds();
    // All constraints (defects + algebraic) must be equality [0, 0].
    for (std::size_t i = 0; i < gl.size(); ++i) {
        EXPECT_DOUBLE_EQ(gl[i], 0.0) << "constraint lower bound at index " << i;
        EXPECT_DOUBLE_EQ(gu[i], 0.0) << "constraint upper bound at index " << i;
    }
}

TEST(HermiteSimpsonAlgebraic, LayoutHasCorrectAlgebraicStride) {
    const std::size_t num_intervals = 3;
    auto ocp = make_twice_x_algebraic_ocp(3.0, num_intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_alg_stride");
    // ns=1, nc=0, na=1, nn=4. stride=2. total_variables=8.
    EXPECT_EQ(compiled.layout.num_algebraic(), 1u);
    EXPECT_EQ(compiled.layout.variables_per_node(), 2u);
    EXPECT_EQ(compiled.layout.total_variables(), 8u);
    // Verify algebraic_index positioning.
    EXPECT_EQ(compiled.layout.algebraic_index(0, 0), 1u);  // node 0: x=0, alg=1
    EXPECT_EQ(compiled.layout.algebraic_index(1, 0), 3u);  // node 1: x=2, alg=3
}

TEST(HermiteSimpsonAlgebraic, AlgebraicBoundsSatisfied) {
    // Bounds test: algebraic variable bounds [-1e19, 1e19] should be set on alg slots.
    const std::size_t num_intervals = 3;
    auto ocp = make_twice_x_algebraic_ocp(3.0, num_intervals);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_alg_varbounds");
    const auto& zl = compiled.problem->variable_lower_bounds();
    const auto& zu = compiled.problem->variable_upper_bounds();
    const std::size_t nn = num_intervals + 1;
    for (std::size_t k = 0; k < nn; ++k) {
        const std::size_t alg_idx = compiled.layout.algebraic_index(k, 0);
        EXPECT_DOUBLE_EQ(zl[alg_idx], -1e19);
        EXPECT_DOUBLE_EQ(zu[alg_idx],  1e19);
    }
}

TEST(HermiteSimpsonAlgebraic, ZeroAlgebraicPreservesExistingBehavior) {
    // Passing a two-template-param OcpProblem must still work exactly as before.
    const double x0 = 1.0, tf = 1.0;
    auto ocp = goss::transcription::test::make_exponential_decay(x0, tf, 10);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "hs_alg_compat");
    EXPECT_EQ(compiled.layout.num_algebraic(), 0u);
    // num_constraints must still be ni * ns = 10 * 1 = 10.
    EXPECT_EQ(compiled.problem->num_constraints(), 10u);
}
```

- [ ] **Step 2: Run to verify tests fail**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_transcription_tests 2>&1 | tail -30'
```

Expected: compile errors or test failures — `algebraic_index` on `VariableLayout` compiled in Task 2 (should build), but the constraint counts will be wrong because the algebraic injection loop does not exist yet. If Task 2 is properly merged, compile errors are replaced by assertion failures in constraint count tests.

- [ ] **Step 3: Extend `HermiteSimpson::compile` in `include/goss/transcription/hermite_simpson.hpp`**

The primary non-uniform overload must be updated. Changes are minimal and localized to three places inside the function body:

**3a. Update VariableLayout construction** (one line change near the top of `compile`):

```cpp
// Old:
VariableLayout layout(ns, nc, nn);

// New:
const std::size_t na = ocp.num_algebraic;
// Validate algebraic bound vectors before touching any element.
if (na > 0) {
    if (ocp.algebraic_lower_bounds.size() != na || ocp.algebraic_upper_bounds.size() != na) {
        throw TranscriptionError(
            "compile: algebraic_lower_bounds and algebraic_upper_bounds "
            "must each have size == num_algebraic");
    }
}
VariableLayout layout(ns, nc, nn, na);
```

**3b. Update `outputs.reserve` and add the algebraic variable extractor helper inside the packed functor:**

The packed functor currently starts with:
```cpp
auto packed = [ocp, layout, ns, nc, ni, node_times](const auto& z) {
    using T = typename std::decay_t<decltype(z)>::value_type;
    std::vector<T> outputs;
    outputs.reserve(1 + ni * ns);
```

Replace with:
```cpp
auto packed = [ocp, layout, ns, nc, ni, na, node_times](const auto& z) {
    using T = typename std::decay_t<decltype(z)>::value_type;
    std::vector<T> outputs;
    // Reserve: 1 cost + ni*ns defects + nn*na algebraic residuals.
    // nn = ni + 1 (one more node than intervals).
    const std::size_t nn_local = ni + 1;
    outputs.reserve(1 + ni * ns + nn_local * na);

    // Helper lambdas to extract x_k, u_k from z.
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
    // Extract the algebraic variable vector at a node. Returns empty vector when na==0.
    auto algebraic_at = [&](std::size_t node) {
        std::vector<T> alg_vars(na);
        for (std::size_t k = 0; k < na; ++k)
            alg_vars[k] = z[layout.algebraic_index(node, k)];
        return alg_vars;
    };
    auto midpoint_control = [&](const std::vector<T>& uk, const std::vector<T>& uk1) {
        std::vector<T> um(nc);
        for (std::size_t j = 0; j < nc; ++j) um[j] = T(0.5) * (uk[j] + uk1[j]);
        return um;
    };
```

**3c. After `for (auto& d : defects) outputs.push_back(d);`, add the algebraic residual loop:**

```cpp
            // Pack outputs: cost first, then defects (same order as before), then
            // algebraic residuals at each node (one per algebraic variable per node).
            outputs.push_back(cost);
            for (auto& d : defects) outputs.push_back(d);

            // Algebraic residual constraints: g(x_k, u_k, alg_k, t_k) == 0 at every node k.
            // One equality constraint per algebraic variable per node.
            // These are added AFTER the defect rows so existing constraint indexing is unchanged.
            if (na > 0) {
                for (std::size_t k = 0; k < nn_local; ++k) {
                    T tk = T(node_times[k]);
                    auto residuals_at_k = ocp.algebraic_residuals_functor(
                        state_at(k), control_at(k), algebraic_at(k), tk);
                    // residuals_at_k has size na; push each residual as a separate constraint row.
                    for (std::size_t j = 0; j < na; ++j) {
                        outputs.push_back(residuals_at_k[j]);
                    }
                }
            }
            return outputs;
```

**3d. Extend variable bounds for algebraic slots** (after the block that pins fixed boundary states):

```cpp
        // Algebraic variable bounds: per-node, same bound for every node.
        // (Algebraic variables are not pinned at boundary nodes — they are free to vary
        // as long as the residual constraint is satisfied.)
        if (na > 0) {
            for (std::size_t k = 0; k < nn; ++k) {
                for (std::size_t j = 0; j < na; ++j) {
                    const std::size_t alg_idx = layout.algebraic_index(k, j);
                    zl[alg_idx] = ocp.algebraic_lower_bounds[j];
                    zu[alg_idx] = ocp.algebraic_upper_bounds[j];
                }
            }
        }
```

**3e. Extend constraint bounds `gl`/`gu`** (replace the existing `num_defects` lines):

```cpp
        // Constraint bounds: defect rows are equalities [0,0]; algebraic residual rows
        // are also equalities [0,0] (the solver enforces g == 0 at every collocation node).
        const std::size_t num_defects = ni * ns;
        const std::size_t num_alg_constraints = nn * na;   // one per algebraic per node
        const std::size_t total_constraints = num_defects + num_alg_constraints;
        std::vector<double> gl(total_constraints, 0.0), gu(total_constraints, 0.0);
```

- [ ] **Step 4: Run to verify tests pass**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_transcription_tests && ctest --test-dir build --output-on-failure -R "HermiteSimpson" 2>&1 | tail -50'
```

Expected: all `HermiteSimpsonAlgebraic.*` tests pass. All pre-existing `HermiteSimpson.*` tests pass unchanged.

- [ ] **Step 5: Run all transcription tests to confirm no regressions**

```bash
scripts/dev.sh 'ctest --test-dir build --output-on-failure 2>&1 | tail -40'
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add include/goss/transcription/hermite_simpson.hpp tests/transcription/test_hermite_simpson.cpp
git commit -m "feat(transcription): extend HermiteSimpson::compile to inject DAE algebraic residual constraints per node"
```

---

## Task 4: `ComposedModel::build` Algebraic Overload and End-to-End DAE Accuracy Test

**Pre-condition:** The accuracy validation suite (`goss_accuracy_tests` target, `tests/accuracy/accuracy_helpers.hpp`) MUST be merged and passing before this task is executed. The end-to-end DAE solve test in this task adds to `tests/accuracy/test_dae_accuracy.cpp` and depends on `accuracy::solve_and_extract_trajectory`.

**Files:**
- Modify: `include/goss/model/composed_model.hpp`
- Modify: `tests/model/test_composed_model.cpp`
- Create: `tests/accuracy/test_dae_accuracy.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `Component::algebraic_entries()`, `Component::num_algebraic()`, `Component::evaluate_algebraic_residual()`, `AlgebraicHandle`, `AlgebraicEntry`, `OcpProblem<Dyn, Cost, AlgRes>`, `Model::build`, `HermiteSimpson::compile`, `IpoptSolver::solve`, `accuracy::solve_and_extract_trajectory`
- Produces:
  - `ComposedModel::total_num_algebraic() const` → `std::size_t`
  - `ComposedModel::validate_algebraic_dimensions() const` (called from `prepare_build`)
  - New `build` overload: `template <typename AlgResFn, typename Dyn0Fn, typename CostFn> auto build(AlgResFn algebraic_residuals, Dyn0Fn component_0_dyn, CostFn combined_cost)`
  - New `build` overload: `template <typename AlgResFn, typename DerivedExpr0Fn, typename Dyn0Fn, typename CostFn> auto build(AlgResFn algebraic_residuals, DerivedExpr0Fn derived_expr_0, Dyn0Fn component_0_dyn, CostFn combined_cost)`
  - Extended `build_internal_model` overload that populates algebraic fields of `OcpProblem`

**DAE test problem design:**

We use a simple scalar semi-explicit index-1 DAE with a known analytic solution:

```
Differential equation:  dx/dt = -x + z_alg
Algebraic constraint:   g(x, u, z_alg, t) = z_alg - c * x = 0   =>  z_alg = c * x
Combined:               dx/dt = -x + c*x = (c-1)*x

With x(0) = 1.0, c = 3.0:
  dx/dt = 2*x   =>  x(t) = exp(2t)
  z_alg(t) = 3 * exp(2t)

Optimal control: minimize integral(x^2) dt over [0, 0.5] with no control input (u = nothing).
The problem has a unique trajectory since x(0) = 1.0 is pinned and there is no control.
Reference values:
  x(0.5) = exp(1.0) ≈ 2.71828182845904523536...
  z_alg(0.5) = 3 * exp(1.0) ≈ 8.15484548537713570608...
  At every node k: z_alg(t_k) / x(t_k) = 3.0 exactly (algebraic constraint satisfaction)
Tolerance: 1e-6 for x(tf) and z_alg(tf); 1e-8 for max |z_alg(t_k) - 3*x(t_k)| across nodes.
```

This problem is:
- **Index-1 semi-explicit DAE** (standard for collocation solvers).
- **Analytically exact** — no numerical approximation of the reference.
- **Trivially verifiable** — `z_alg / x = 3.0` at every node is a direct ratio check.
- **No control** — removes solver ambiguity; the unique trajectory tests DAE correctness in isolation.

- [ ] **Step 1: Write failing unit tests for ComposedModel algebraic metadata collection**

Add to `tests/model/test_composed_model.cpp`:

```cpp
#include "goss/model/component.hpp"
#include "goss/model/composed_model.hpp"
#include "goss/model/errors.hpp"
#include <gtest/gtest.h>
#include <functional>
#include <vector>

TEST(ComposedModelAlgebraic, TotalNumAlgebraicSumsComponents) {
    goss::model::ComposedModel composed;
    goss::model::Component component_a("comp_a");
    component_a.add_state("x");
    auto alg_fn_a = [](const std::vector<double>&, const std::vector<double>&,
                       const std::vector<double>&, double) -> double { return 0.0; };
    component_a.add_algebraic("z1", alg_fn_a, -1e19, 1e19);
    goss::model::Component component_b("comp_b");
    auto alg_fn_b = [](const std::vector<double>&, const std::vector<double>&,
                       const std::vector<double>&, double) -> double { return 0.0; };
    // Note: comp_b has no states — but for this unit test we only check counting.
    // The full build() guards are tested in the solve test.
    composed.add_component(std::move(component_a));
    // comp_b has no states so we won't call build(), but total_num_algebraic should still work.
    EXPECT_EQ(composed.total_num_algebraic(), 1u);
}

TEST(ComposedModelAlgebraic, TotalNumAlgebraicZeroWhenNoneRegistered) {
    goss::model::ComposedModel composed;
    goss::model::Component component("comp");
    component.add_state("position");
    composed.add_component(std::move(component));
    EXPECT_EQ(composed.total_num_algebraic(), 0u);
}
```

- [ ] **Step 2: Run to verify tests fail**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests 2>&1 | tail -20'
```

Expected: compile error — `total_num_algebraic` not a member of `ComposedModel`.

- [ ] **Step 3: Add `total_num_algebraic()` and `validate_algebraic_dimensions()` to ComposedModel**

In `include/goss/model/composed_model.hpp`, add after `total_num_derived()`:

```cpp
    /// Count of algebraic variables across all registered components.
    std::size_t total_num_algebraic() const {
        std::size_t total = 0;
        for (const auto& component : components_) {
            total += component.num_algebraic();
        }
        return total;
    }
```

Add a `validate_algebraic_dimensions()` private method (analogous to `validate_dynamics_dimensions`):

```cpp
    /// For each component that has algebraic entries, verify that
    /// evaluate_algebraic_residual returns the expected scalar value without throwing.
    /// Uses zero-filled probe vectors of correct global sizes.
    void validate_algebraic_dimensions() const {
        const std::size_t num_states       = total_num_states();
        const std::size_t num_algebraics   = total_num_algebraic();
        const std::vector<double> probe_x(num_states, 0.0);
        const std::vector<double> probe_u(control_names_.size(), 0.0);
        const std::vector<double> probe_alg(num_algebraics, 0.0);
        constexpr double probe_t = 0.0;
        for (const auto& component : components_) {
            for (std::size_t j = 0; j < component.num_algebraic(); ++j) {
                // evaluate_algebraic_residual must not throw with zero-filled inputs.
                // It returns a scalar; we only check it doesn't throw (dimension mismatch
                // inside the lambda would be a runtime error caught here).
                component.evaluate_algebraic_residual(j, probe_x, probe_u, probe_alg, probe_t);
            }
        }
    }
```

Call `validate_algebraic_dimensions()` inside `prepare_build()` after `validate_dynamics_dimensions()`:

```cpp
        validate_dynamics_dimensions();
        validate_algebraic_dimensions();
```

- [ ] **Step 4: Add `build` overload for algebraic problems with 0 inline derived quantities**

In `include/goss/model/composed_model.hpp`, add a new `build` overload after the existing 0-derived overload. This overload takes an `AlgResFn` generic lambda as its first argument:

**Convention for this overload's argument ordering:**
1. Algebraic residuals generic lambda: `(const auto& x, const auto& u, const auto& alg_vars, auto t) -> vector<T>` of size `num_algebraic`.
2. Component 0 dynamics generic lambda: same as existing build.
3. Combined cost generic lambda: same as existing build.

```cpp
    /// Build overload for 0 inline derived quantities + 1 algebraic residual functor.
    ///
    /// AD-safety: AlgResFn must be a concrete generic lambda (not std::function).
    /// It is captured by value into the packed functor in build_internal_model_alg()
    /// and called during CppADCG recording under AD types.
    template <typename AlgResFn, typename Dyn0Fn, typename CostFn>
    auto build_with_algebraic(AlgResFn algebraic_residuals,
                               Dyn0Fn component_0_dyn,
                               CostFn combined_cost) {
        prepare_build();

        // Guard: this overload is for 0 inline derived quantities.
        {
            const std::size_t nd = total_num_derived();
            if (nd != 0) {
                throw ComponentError(
                    "ComposedModel::build_with_algebraic (0-derived overload): "
                    "0 inline derived quantities expected, found " + std::to_string(nd) +
                    "; use the 1-derived algebraic overload.");
            }
        }

        const std::size_t num_states      = total_num_states();
        const std::size_t num_alg         = total_num_algebraic();

        std::vector<std::size_t> component_state_offsets = compute_component_state_offsets();
        std::size_t comp0_offset  = 0;
        std::size_t comp0_nstates = 0;
        for (std::size_t ci = 0; ci < components_.size(); ++ci) {
            if (components_[ci].num_owned_states() > 0) {
                comp0_offset  = component_state_offsets[ci];
                comp0_nstates = components_[ci].num_owned_states();
                break;
            }
        }

        // Combined dynamics: ignores the algebraic variables — they are NLP decision
        // variables, not arguments to the ODE dynamics. The ODE dynamics may READ
        // algebraic variable values from x (they are accessible as states if the caller
        // wires them that way), but typically the dynamics read the alg_vars via the
        // component_0_dyn signature (which takes the full x and deriveds vectors).
        //
        // IMPORTANT: component_0_dyn takes (x, u, deriveds, t). deriveds is empty here
        // (0 inline derived quantities). The algebraic variable vector is separate from
        // x and is not passed into dynamics — the ODE sees only x and u. For DAE problems
        // where the dynamics need z_alg, they should read it via an input_state or the
        // combined combined dynamics lambda should extract it from the flat z vector.
        //
        // For the canonical semi-explicit index-1 DAE:
        //   dx/dt = f(x, z_alg)
        //   g(x, z_alg) = 0
        // The caller's component_0_dyn lambda receives the global x vector which does NOT
        // include z_alg slots (those are separate NLP variables). To pass z_alg into f,
        // the caller should pass a dynamics lambda that accepts all z (including alg slots)
        // extracted from a broader context, or register algebraic variables as states.
        //
        // Simplest pattern: pass algebraic values through an extra "algebraic state" slot
        // in x; or write dynamics as f(x, u, t) where the alg_vars are implicit (the
        // solver guarantees g=0 so the dynamics and residual are consistent post-solve).
        // The test below uses the implicit pattern: dynamics = -x + z_alg, but since
        // z_alg = c*x (enforced by g=0), the dynamics lambda simply computes (c-1)*x
        // using the x slot directly, trusting the solver to enforce consistency.
        //
        // This avoids needing to thread alg_vars into the dynamics signature — a variadic
        // extension that reads alg slots from z inside dynamics is a future follow-on.
        auto combined_dynamics = [
            component_0_dyn,
            num_states,
            comp0_offset,
            comp0_nstates
        ](const auto& x, const auto& u, auto t) {
            using T = typename std::decay_t<decltype(x)>::value_type;
            std::vector<T> deriveds;
            std::vector<T> dx(num_states);
            auto comp0_dx = component_0_dyn(x, u, deriveds, t);
            for (std::size_t i = 0; i < comp0_nstates; ++i) {
                dx[comp0_offset + i] = comp0_dx[i];
            }
            return dx;
        };

        auto ocp_cost = [combined_cost](const auto& x, const auto& u, auto t) {
            using T = typename std::decay_t<decltype(x)>::value_type;
            std::vector<T> deriveds;
            return combined_cost(x, u, deriveds, t);
        };

        return build_internal_model_alg(
            std::move(combined_dynamics), std::move(ocp_cost),
            std::move(algebraic_residuals), num_alg);
    }
```

Add `build_internal_model_alg` private method (extends `build_internal_model` to carry algebraic metadata):

```cpp
    /// Extend build_internal_model to populate algebraic fields of OcpProblem.
    /// Collects algebraic bounds from all components and forwards them through
    /// the transcription pipeline alongside the dynamics and cost functors.
    template <typename DynamicsFn, typename CostFn, typename AlgResFn>
    auto build_internal_model_alg(DynamicsFn dynamics, CostFn cost,
                                   AlgResFn algebraic_residuals,
                                   std::size_t num_alg) const {
        Model internal_model;

        for (const auto& comp : components_) {
            for (const auto& owned_state : comp.owned_states()) {
                auto sh = internal_model.add_state(owned_state.name);
                internal_model.set_state_bounds(sh, owned_state.lower_bound, owned_state.upper_bound);
                if (owned_state.initial_fixed) {
                    internal_model.set_initial_state(sh, owned_state.initial_value);
                }
                if (owned_state.final_fixed) {
                    internal_model.set_final_state(sh, owned_state.final_value);
                }
            }
        }

        for (std::size_t ci = 0; ci < control_names_.size(); ++ci) {
            auto ch = internal_model.add_control(control_names_[ci]);
            internal_model.set_control_bounds(ch, control_lower_[ci], control_upper_[ci]);
        }

        internal_model.set_mesh(mesh_.t_initial, mesh_.t_final, mesh_.num_intervals);

        // Build the base OcpProblem (two-template-param form).
        auto base_ocp = internal_model.build(std::move(dynamics), std::move(cost));

        // Collect algebraic bounds from all components in registration order.
        // Components lay out their algebraic variables contiguously in the same order.
        std::vector<double> alg_lower_bounds;
        std::vector<double> alg_upper_bounds;
        alg_lower_bounds.reserve(num_alg);
        alg_upper_bounds.reserve(num_alg);
        for (const auto& comp : components_) {
            for (const auto& alg_entry : comp.algebraic_entries()) {
                alg_lower_bounds.push_back(alg_entry.lower_bound);
                alg_upper_bounds.push_back(alg_entry.upper_bound);
            }
        }

        // Construct the three-template-param OcpProblem with algebraic fields.
        // C++17 aggregate initialization: list all positional members in struct order.
        using BaseDynFn  = typename decltype(base_ocp)::dynamics_fn_type;
        // Note: OcpProblem does not expose dynamics_fn_type as a typedef — we use the
        // decltype of the base_ocp directly and build a new OcpProblem<Dyn, Cost, Alg>.
        //
        // Since OcpProblem is a struct template, we cannot extract the Dyn/Cost types
        // from base_ocp without a typedef. Instead, use template argument deduction:
        // pass the components individually to a helper that constructs the 3-param form.
        return make_algebraic_ocp(std::move(base_ocp), std::move(algebraic_residuals),
                                  num_alg,
                                  std::move(alg_lower_bounds),
                                  std::move(alg_upper_bounds));
    }

    /// Helper to construct OcpProblem<Dyn, Cost, AlgRes> from a base OcpProblem<Dyn, Cost>
    /// and algebraic metadata, without naming the Dyn/Cost types explicitly.
    template <typename DynamicsFn, typename CostFn, typename AlgResFn>
    static auto make_algebraic_ocp(
            transcription::OcpProblem<DynamicsFn, CostFn> base_ocp,
            AlgResFn algebraic_residuals,
            std::size_t num_alg,
            std::vector<double> alg_lower_bounds,
            std::vector<double> alg_upper_bounds) {
        transcription::OcpProblem<DynamicsFn, CostFn, AlgResFn> ocp{
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
            num_alg,
            std::move(algebraic_residuals),
            std::move(alg_lower_bounds),
            std::move(alg_upper_bounds)
        };
        return ocp;
    }
```

- [ ] **Step 5: Run unit tests to verify ComposedModel algebraic metadata tests pass**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_model_tests && ctest --test-dir build --output-on-failure -R "ComposedModelAlgebraic" 2>&1 | tail -30'
```

Expected: `ComposedModelAlgebraic.*` tests pass. All pre-existing model tests pass.

- [ ] **Step 6: Write the end-to-end DAE accuracy test**

**The DAE test problem (full specification):**

Semi-explicit index-1 DAE optimal control:
```
States:        x(t) ∈ ℝ  (1 differential variable)
Controls:      (none)
Algebraic:     z_alg(t) ∈ ℝ  (1 algebraic variable)
ODE:           dx/dt = -x + z_alg        ... (A)
Algebraic eq:  g(x, z_alg) = z_alg - c*x = 0  (enforces z_alg = c*x)  ... (B)
Cost:          0  (feasibility problem — unique trajectory)
Horizon:       t ∈ [0, 0.5]
IC:            x(0) = 1.0  (fixed)
Parameter:     c = 3.0

Substituting (B) into (A):
  dx/dt = -x + 3x = 2x  =>  x(t) = exp(2t)
  z_alg(t) = 3*exp(2t)

Reference values:
  x(0.5)       = exp(1.0) ≈ 2.718281828459045
  z_alg(0.5)   = 3*exp(1.0) ≈ 8.154845485377135
  At every node k: z_alg_k / x_k = 3.0 exactly
```

The DAE is index-1 because the Jacobian ∂g/∂z_alg = 1 is non-singular everywhere. The problem has a unique solution since x(0) is pinned and the ODE is autonomous. There is no control, so the NLP is purely a feasibility problem with a trivial zero cost.

**NOTE on dynamics lambda:** Because `build_with_algebraic` does not thread `alg_vars` into the dynamics signature (see the comment in the overload above), the dynamics lambda for this test must be written as if `z_alg = c*x` is already known (i.e., the combined substituted ODE `dx/dt = (c-1)*x`). The algebraic residual constraint independently forces the solver to satisfy `z_alg = c*x` at every node. This is the canonical "implicit DAE" usage pattern: the dynamics and constraint together define the system; the solver enforces both simultaneously.

Create `tests/accuracy/test_dae_accuracy.cpp`:

```cpp
// tests/accuracy/test_dae_accuracy.cpp
//
// End-to-end accuracy test for DAE Flavor 2 (algebraic-variable) derived quantities.
//
// Problem: semi-explicit index-1 DAE on [0, 0.5].
//   dx/dt = -x + z_alg
//   g(x, z_alg) = z_alg - 3*x = 0   =>  z_alg = 3*x
//
// Substituted ODE (for the dynamics functor, since alg_vars are not threaded
// into dynamics in v1): dx/dt = 2*x
//
// Analytic solution:
//   x(t)     = exp(2*t)
//   z_alg(t) = 3*exp(2*t)
//
// Assertions:
//   1. x(tf) is within 1e-6 of exp(1.0).
//   2. z_alg(tf) is within 1e-6 of 3*exp(1.0).
//   3. At every node k: |z_alg_k / x_k - 3.0| < 1e-8
//      (algebraic constraint g == 0 is satisfied to tight tolerance at every node).
//
// Pre-condition: accuracy validation suite (goss_accuracy_tests) must be merged before
// this file is compiled. It provides accuracy::solve_and_extract_trajectory.
#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <vector>
#include "goss/model/component.hpp"
#include "goss/model/composed_model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "accuracy/accuracy_helpers.hpp"

namespace {

// The algebraic residual g(x, u, z_alg, t) = z_alg[0] - 3.0 * x[0].
// Enforces z_alg[0] = 3 * x[0] at every collocation node when driven to zero.
struct ThreeTimesXResidual {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x,
                               const std::vector<T>& /*u*/,
                               const std::vector<T>& alg_vars,
                               T /*t*/) const {
        return { alg_vars[0] - T(3.0) * x[0] };
    }
};

}  // namespace

TEST(DaeAccuracy, SemiExplicitIndex1DaeTracksAnalyticTrajectory) {
    // ---- Problem parameters ----
    constexpr double x_initial        = 1.0;
    constexpr double time_horizon     = 0.5;
    constexpr double algebraic_coeff  = 3.0;  // z_alg = algebraic_coeff * x
    constexpr std::size_t num_intervals = 40;

    // ---- Analytic reference values ----
    // x(t) = exp(2*t), z_alg(t) = 3*exp(2*t)
    const double x_final_reference      = std::exp(2.0 * time_horizon);  // exp(1.0)
    const double z_alg_final_reference  = algebraic_coeff * x_final_reference;

    // ---- Model assembly ----
    goss::model::ComposedModel composed;

    goss::model::Component dae_component("dae_system");
    const auto state_handle = dae_component.add_state("x");
    dae_component.set_initial_state(state_handle, x_initial);
    dae_component.set_state_bounds(state_handle, -1e19, 1e19);

    // Algebraic variable z_alg with bounds [-1e19, 1e19] (unconstrained in box sense;
    // the residual constraint z_alg = 3*x will fully determine its value).
    auto algebraic_validation_fn = [algebraic_coeff](
            const std::vector<double>& x,
            const std::vector<double>& /*u*/,
            const std::vector<double>& alg_vars,
            double /*t*/) -> double {
        return alg_vars[0] - algebraic_coeff * x[0];
    };
    dae_component.add_algebraic("z_alg", algebraic_validation_fn, -1e19, 1e19);

    // Dynamics: dx/dt = 2*x (substituted form; z_alg = 3*x enforced separately by residual).
    // Note: the dynamics functor does not receive alg_vars in v1; the caller must write
    // the dynamics in the substituted form for a semi-explicit DAE.
    dae_component.set_dynamics(
        [](const std::vector<double>& x,
           const std::vector<double>& /*u*/,
           const std::vector<double>& /*deriveds*/,
           double /*t*/) -> std::vector<double> {
            // dx/dt = (c-1)*x where c=3 => 2*x
            return { 2.0 * x[0] };
        });
    composed.add_component(std::move(dae_component));
    composed.set_mesh(0.0, time_horizon, num_intervals);

    // ---- AD-safe generic lambdas for build_with_algebraic ----

    // Algebraic residuals functor (AD-safe, concrete type ThreeTimesXResidual).
    // This is the SAME computation as the validation lambda above but as a generic lambda
    // (template operator()) so it can be called under CppAD AD scalar types during recording.
    auto algebraic_residuals = ThreeTimesXResidual{};

    // Component dynamics (generic lambda, AD-safe).
    // dx/dt = 2*x (substituted DAE — z_alg = 3*x enforced by algebraic constraint).
    auto dae_dynamics = [](const auto& x, const auto& /*u*/, const auto& /*deriveds*/, auto /*t*/) {
        using ScalarT = typename std::decay_t<decltype(x)>::value_type;
        return std::vector<ScalarT>{ ScalarT(2.0) * x[0] };
    };

    // Zero cost (feasibility problem — unique trajectory determined by IC + DAE).
    auto zero_cost = [](const auto& /*x*/, const auto& /*u*/,
                        const auto& /*deriveds*/, auto /*t*/) {
        using ScalarT = typename std::decay_t<decltype(x_initial)>::value_type;
        // Workaround for unused type in generic lambda: deduce T from the argument.
        // Since u is empty and x is passed as const auto&, we use double here safely.
        return 0.0;
    };

    // ---- Solve ----
    auto ocp = composed.build_with_algebraic(algebraic_residuals, dae_dynamics, zero_cost);
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "dae_accuracy_test");

    // Initial guess: flat z0 = 1.0 for all variables (states and algebraic variables).
    // IPOPT will satisfy the algebraic constraints from this guess.
    std::vector<double> initial_guess(compiled.problem->num_variables(), 1.0);

    goss::solver::IpoptSolver solver;
    solver.set_tolerance(1e-10);  // tight solver tolerance so algebraic constraint error is below 1e-8
    const auto result = solver.solve(*compiled.problem, initial_guess);

    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success)
        << "IPOPT failed to converge on the semi-explicit index-1 DAE OCP";

    const auto& layout = compiled.layout;
    const std::size_t nn = layout.num_nodes();

    // ---- Assertion 1: x(tf) matches analytic reference ----
    const double x_final_computed =
        result.x[layout.state_index(nn - 1, 0)];
    EXPECT_NEAR(x_final_computed, x_final_reference, 1e-6)
        << "x(tf) = " << x_final_computed
        << " but analytic reference is exp(1.0) = " << x_final_reference;

    // ---- Assertion 2: z_alg(tf) matches analytic reference ----
    const double z_alg_final_computed =
        result.x[layout.algebraic_index(nn - 1, 0)];
    EXPECT_NEAR(z_alg_final_computed, z_alg_final_reference, 1e-6)
        << "z_alg(tf) = " << z_alg_final_computed
        << " but analytic reference is 3*exp(1.0) = " << z_alg_final_reference;

    // ---- Assertion 3: algebraic constraint z_alg = 3*x satisfied at every node ----
    // At every collocation node k, the residual g(x_k, z_alg_k) = z_alg_k - 3*x_k
    // must be zero to within 1e-8 (solver tolerance is 1e-10; discretization error
    // is O(h^4) for HermiteSimpson with 40 intervals over [0,0.5], well below 1e-8).
    for (std::size_t k = 0; k < nn; ++k) {
        const double x_k     = result.x[layout.state_index(k, 0)];
        const double z_alg_k = result.x[layout.algebraic_index(k, 0)];
        const double residual_k = z_alg_k - algebraic_coeff * x_k;
        EXPECT_NEAR(residual_k, 0.0, 1e-8)
            << "Algebraic constraint g = z_alg - 3*x violated at node " << k
            << ": z_alg=" << z_alg_k << ", x=" << x_k
            << ", residual=" << residual_k;
    }

    // ---- Assertion 4: x is non-decreasing (monotone for dx/dt = 2*x with x(0)>0) ----
    for (std::size_t k = 1; k < nn; ++k) {
        EXPECT_GE(result.x[layout.state_index(k, 0)],
                  result.x[layout.state_index(k - 1, 0)] - 1e-6)
            << "x should be non-decreasing (dx/dt=2x, x(0)=1 > 0) at node " << k;
    }
}
```

- [ ] **Step 7: Wire `test_dae_accuracy.cpp` into CMakeLists.txt**

In `CMakeLists.txt`, find the `goss_accuracy_tests` executable target (added by the accuracy validation suite plan). Add the new file to its sources list:

```cmake
add_executable(goss_accuracy_tests
  tests/accuracy/test_closed_form.cpp
  tests/accuracy/test_benchmarks.cpp
  tests/accuracy/test_convergence_order.cpp
  tests/accuracy/test_invariants.cpp
  tests/accuracy/test_dae_accuracy.cpp)   # <-- add this line
```

(The exact sources list depends on what the accuracy suite plan created; add `tests/accuracy/test_dae_accuracy.cpp` to whatever list is present for `goss_accuracy_tests`.)

- [ ] **Step 8: Run to verify tests fail (accuracy suite not yet merged)**

If the accuracy suite is not yet merged, this step will fail at `#include "accuracy/accuracy_helpers.hpp"`. This is expected — the pre-condition says the accuracy suite must merge first. If the accuracy suite IS already merged:

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_accuracy_tests 2>&1 | tail -30'
```

Expected (pre-accuracy-suite merge): compile error on `accuracy/accuracy_helpers.hpp`.
Expected (post-accuracy-suite merge, before this task's model changes): compile error on `build_with_algebraic` not found.

- [ ] **Step 9: Run the full test suite to confirm all tests pass**

```bash
scripts/dev.sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --output-on-failure 2>&1 | tail -60'
```

Expected: all targets build, all tests pass including `DaeAccuracy.SemiExplicitIndex1DaeTracksAnalyticTrajectory`.

- [ ] **Step 10: Commit**

```bash
git add include/goss/model/composed_model.hpp \
        tests/model/test_composed_model.cpp \
        tests/accuracy/test_dae_accuracy.cpp \
        CMakeLists.txt
git commit -m "feat(model): add ComposedModel::build_with_algebraic and DAE end-to-end accuracy test for Flavor 2 algebraic variables"
```

---

## Self-Review

### 1. Spec Coverage — 4 Design Steps from component.hpp Lines 159–179

| Design Step | Covered In |
|---|---|
| Step 1: OcpProblem gains `algebraic_residuals` functor + `num_algebraic` count + per-variable bounds | Task 2 (`OcpProblem` extension) |
| Step 2: VariableLayout extension — algebraic variables become per-node slots, `algebraic_index(node, j)` accessor | Task 2 (`VariableLayout` extension) |
| Step 3: Transcription defect seam — one algebraic residual constraint per algebraic variable per collocation node, added to gl/gu as equality (=0) constraints | Task 3 (`HermiteSimpson::compile` extension). **Trapezoidal and LGL are explicitly deferred** per the scope decision in Global Constraints. |
| Step 4: ComposedModel::build() extension — collect algebraic entries from all components, populate OcpProblem::algebraic_residuals + num_algebraic | Task 4 (`ComposedModel::build_with_algebraic` and `build_internal_model_alg`) |

### 2. Placeholder Scan

No "TBD", "TODO", "implement later", or "fill in details" found. All code blocks contain actual signatures, actual formulas, and actual test logic.

### 3. VariableLayout Index-Formula Audit for Regressions

The new `variables_per_node()` is `ns + nc + na`. For existing callers with `na = 0`:
- `state_index(node, i) = node * (ns + nc + 0) + i = node * (ns + nc) + i` ✓ (same as before)
- `control_index(node, j) = node * (ns + nc + 0) + ns + j = node * (ns + nc) + ns + j` ✓ (same as before)
- `total_variables() = num_nodes * (ns + nc + 0) = num_nodes * (ns + nc)` ✓ (same as before)

For `na > 0`, algebraic slots are appended at offset `ns + nc + k` within each node group, so they never collide with state or control slots.

The existing unit tests in `test_variable_layout.cpp` (`StateAndControlIndicesAreNodeGrouped`, `ZeroControlsIsAllowed`) both use the 3-arg constructor which now defaults `num_algebraic = 0` — all existing index values remain identical.

### 4. AD-Safety Audit

- `AlgebraicEntry::validation_fn` is `std::function<double(...)>` — used ONLY in `Component::evaluate_algebraic_residual` for validation-path probing in `validate_algebraic_dimensions()`. Never stored in or called from a packed functor.
- `ThreeTimesXResidual` and the `algebraic_residuals` generic lambda in test and model code are concrete types (struct with `template<T> operator()`) captured by value into `ocp.algebraic_residuals_functor`. The packed functor inside `HermiteSimpson::compile` captures `ocp` by value (which captures `AlgResFn algebraic_residuals_functor` by value). When CppADCG records over `CppAD::AD<CppAD::cg::CG<double>>` scalar type `T`, `ocp.algebraic_residuals_functor(xk, uk, alg_k, tk)` instantiates the concrete `operator()<T>` — no `std::function` in the AD path.
- `build_with_algebraic` takes `AlgResFn` as a concrete template parameter; it is never assigned to `std::function` before being passed to `build_internal_model_alg` → `make_algebraic_ocp` → stored in `OcpProblem::algebraic_residuals_functor`.

### 5. Backward-Compatibility Check

- `OcpProblem<Dyn, Cost>` (two-param) deduction: `AlgResFn = NoAlgebraicResiduals` by default. The three new fields (`num_algebraic = 0`, `algebraic_residuals_functor = AlgResFn{}`, empty bound vectors) have default values in C++17 aggregate initialization, so existing 12-member positional aggregate-init calls compile unchanged.
- `VariableLayout(ns, nc, nn)` (3-arg) call: `num_algebraic = 0` by default. All existing index computations produce identical values.
- `HermiteSimpson::compile` called with a 2-param `OcpProblem`: `ocp.num_algebraic == 0`, so the `if (na > 0)` branch is never taken. `VariableLayout(ns, nc, nn, 0)` is called, identical to the old `VariableLayout(ns, nc, nn)`. `outputs.reserve(1 + ni*ns + 0)` and `gl`/`gu` size is `ni*ns + 0 = ni*ns`. All existing `HermiteSimpson.*` tests pass unchanged.
- Existing `ComposedModel::build(...)` overloads are untouched. Only `build_with_algebraic` is added.

### 6. DAE Accuracy Target Summary

- Problem: semi-explicit index-1 DAE, `dx/dt = 2x`, `g = z_alg - 3x = 0`, `x(0)=1`, `t ∈ [0, 0.5]`.
- Reference: `x(0.5) = exp(1.0)`, `z_alg(0.5) = 3*exp(1.0)`, `z_alg/x = 3.0` at every node.
- Tolerances: `1e-6` on `x(tf)` and `z_alg(tf)`; `1e-8` on per-node algebraic constraint residual.
- Scheme: HermiteSimpson with 40 intervals (O(h^4) ≈ 1.5e-11 discretization error, well inside tolerance).

### 7. Biggest Regression Risk

**The single biggest regression risk is the `variables_per_node()` change in `VariableLayout` being invoked by `HermiteSimpson::compile` with `na = 0` but producing a different value due to a coding error in the default argument.**

Concretely: if `num_algebraic_` were accidentally initialized to a non-zero value when the 3-arg constructor is called (e.g., due to a brace-initialization mistake), all existing `state_index` and `control_index` calls would return wrong values, silently corrupting the NLP. The mitigation is: (a) the 3-arg constructor explicitly passes `0` as the fourth argument of the same member-initializer-list, (b) the new unit test `ZeroAlgebraicPreservesExistingIndices` directly compares `state_index` and `control_index` values from the 3-arg and 4-arg-with-zero constructors, and (c) all pre-existing `HermiteSimpson.*` tests (convergence, pin, boundary) continue to run and serve as regression guards.
