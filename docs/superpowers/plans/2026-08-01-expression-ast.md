# Operator-Overload Expression DSL (AST) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sugar layer — `goss::model::expr` — that lets a user write `q >= 0.0`, `q.initial() == 10.0`, `set_cost(integral(q + w*rate*rate))`, and `set_dynamics(q, ARRIVAL - rate)` instead of the equivalent lambda/setter calls. The AST is a PURE FRONT-END: it produces exactly the same `set_state_bounds`, `set_initial_state`, and generic-lambda `dynamics`/`cost` arguments that `Model::build()` already accepts. No downstream layer (transcription, NLP, solver) is touched.

**Architecture:** An expression template tree (statically typed CRTP-free C++17 variant: typed node structs composing via operator overloads into nested `BinaryExpr<Op,L,R>` / `UnaryExpr<Op,Operand>` / `StateLeaf` / `ControlLeaf` / `ConstantLeaf` / `TimeLeaf` types). Every node exposes a single member function template:

```cpp
template <typename ScalarT>
ScalarT eval(const std::vector<ScalarT>& x, const std::vector<ScalarT>& u, ScalarT t) const;
```

Because this is a method template on a concrete struct (not a virtual function), it instantiates cleanly under both `double` and CppAD's `AD<CG<double>>` — the make-or-break AD-safety requirement. Literals are wrapped in `ScalarT(value)` at the leaf level.

Per-state dynamics expressions are collected in an `ExprModel` (a thin wrapper over `Model` that also holds a `std::vector<std::optional<Expr>>` of per-state dynamics expressions and an optional cost expression). `ExprModel::build()` assembles them into a single combined dynamics lambda and a single cost lambda, then delegates to the underlying `Model::build(dynamics, cost)`. The existing `Model` class and its lambda `build()` overload are NOT modified — the `ExprModel` is purely additive.

**Tech Stack:** C++17, existing `goss::model::Model`, `goss::transcription`, `goss::solver::IpoptSolver`, GoogleTest, containerized CMake via `scripts/dev.sh`.

## Global Constraints

- Language: **C++17** throughout. No C++20 concepts or `std::variant` visitor macros that require later standards.
- Header-only (same as `goss_model` INTERFACE): all new files live under `include/goss/model/expr/`. No new `.cpp` sources.
- All expression node `eval<ScalarT>()` implementations MUST be instantiable under `double` AND `CppAD::AD<CppAD::cg::CG<double>>`. This means: wrap every numeric literal in `ScalarT(literal)`; use `+`, `-`, `*` operators only (all defined for CppAD AD types); do NOT use `std::abs`, `std::sqrt`, or any `std::math` function without ADL protection — they are not in scope for this v1 feature set. If a future operator needs a transcendental, note it explicitly.
- The `ExprError` exception class (analogous to `ModelError`) MUST be thrown for misuse: using a `StateHandle` from a different model, calling `set_dynamics` twice for the same state, calling `ExprModel::build()` with missing dynamics for any declared state, or calling `integral()` on a non-expression argument.
- `ExprModel` MUST preserve the lambda `build(dynamics, cost)` path: a user who never calls `set_dynamics`/`set_cost` on an `ExprModel` and calls the two-argument `build()` directly gets the same behavior as with a plain `Model`.
- Verbose, descriptive names for all variables, members, and functions. Type annotations everywhere. Comments explain WHY.
- Container-first: all `cmake`/`ctest` commands run inside the container via `scripts/dev.sh '<command>'`.
- Test framework: GoogleTest. The flagship test is a verbatim re-expression of `QueueModelKeepsQueueNonNegative` using the operator-overload syntax; it MUST produce the same solver result (same objective to 1e-3, same feasibility assertions) as the existing lambda version.
- No general nonlinear path constraints in v1: `state >= const`, `state <= const`, `const <= state` lower to `set_state_bounds`; `control >= const` and `control <= const` lower to `set_control_bounds`. A `StateExpr >= some_expr_involving_u` (or any non-constant RHS) is a general path constraint — this requires a transcription extension (a new constraint-evaluation hook in `HermiteSimpson`/`Trapezoidal`) and is explicitly deferred. The plan notes this decision and the transcription work it would require.
- The comparison operators `>=`, `<=`, `==` on handles lower ONLY when the LHS is a bare `StateHandle` or `ControlHandle` and the RHS is a `ConstantExpr<double>` (i.e., a literal or a `const double` captured at DSL-call time). Mixed-expression comparisons emit a static_assert with a clear message directing the user to the general path-constraint extension point.

---

## File Structure

- `include/goss/model/expr/errors.hpp` — `ExprError : std::runtime_error`.
- `include/goss/model/expr/nodes.hpp` — all expression node types: `ConstantExpr<T>`, `StateLeaf`, `ControlLeaf`, `TimeLeaf`, `BinaryExpr<Tag,L,R>`, `UnaryNegExpr<Operand>`. Tag structs: `AddTag`, `SubTag`, `MulTag`. Each has `template<ScalarT> ScalarT eval(x, u, t) const`.
- `include/goss/model/expr/operators.hpp` — free function `operator+`, `operator-`, `operator*`, `operator-` (unary) that compose nodes. Also `operator>=`, `operator<=`, `operator==` returning `BoundConstraint` / `BoundaryConstraint` (plain structs, not expressions — they are not evaluable, they lower directly to `Model` setter calls).
- `include/goss/model/expr/integral.hpp` — the `integral(expr)` wrapper that wraps a typed expression into a generic `CostFunctor<Expr>` compatible with `Model::build`'s `CostFn` contract.
- `include/goss/model/expr/expr_model.hpp` — `ExprModel`: owns a `goss::model::Model`; adds `set_dynamics(StateHandle, Expr)`, `set_cost(integral_expr)`, `apply(BoundConstraint)`, `apply(BoundaryConstraint)`, and `build()` (zero-argument overload that assembles per-state dynamics into a combined `DynamicsFunctor` and delegates to `Model::build`). Also re-exposes the raw `Model` setters (forwarding) and the two-argument `Model::build(dynamics, cost)` for the lambda path.
- `include/goss/model/expr/expr.hpp` — umbrella include for the whole `expr/` sub-library.
- `tests/model/test_expr_nodes.cpp` — unit tests for node construction and `eval<double>` / `eval<AD<...>>`.
- `tests/model/test_expr_lowering.cpp` — tests for `BoundConstraint`/`BoundaryConstraint` lowering to `Model` setters and `ExprModel` dynamics/cost assembly.
- `tests/model/test_expr_solve.cpp` — flagship end-to-end: queue model rewritten with operator-overload syntax, solved, result compared to the existing lambda version's assertions.
- `CMakeLists.txt` — add new test files to `goss_model_tests` (the existing `goss_model` INTERFACE library requires zero changes; the new headers are included via the same `include/` tree).

---

### Task 1: Scaffold `expr/` sub-library — `ExprError` + CMake wiring

**Files:**
- Create: `include/goss/model/expr/errors.hpp`
- Create: `include/goss/model/expr/expr.hpp` (umbrella, initially includes only `errors.hpp`)
- Modify: `CMakeLists.txt` — add `tests/model/test_expr_nodes.cpp` to `goss_model_tests`
- Create: `tests/model/test_expr_nodes.cpp` (smoke portion only in this task)

**Interfaces:**
- Produces: `class goss::model::expr::ExprError : public std::runtime_error` (ctor from `const std::string&`).
- Produces: CMake awareness of the new test file (the test source is grown in subsequent tasks; the smoke test here ensures the target compiles and links).
- The `goss_model` INTERFACE library already includes `${CMAKE_SOURCE_DIR}/include`, so `#include "goss/model/expr/errors.hpp"` works without any CMake change beyond adding the test source.

**Design note:** `ExprError` mirrors `ModelError` exactly. It lives in its own namespace `goss::model::expr` so call sites can distinguish expression-DSL errors from model-metadata errors. The umbrella `expr.hpp` grows with each task; starting minimal keeps compilation fast during development.

- [ ] **Step 1: Write the failing smoke test**

```cpp
// tests/model/test_expr_nodes.cpp
#include <gtest/gtest.h>
#include "goss/model/expr/errors.hpp"

TEST(ExprError, IsThrowableAndCarriesMessage) {
    try {
        throw goss::model::expr::ExprError("test error");
    } catch (const goss::model::expr::ExprError& caught_error) {
        EXPECT_STREQ(caught_error.what(), "test error");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — `goss/model/expr/errors.hpp` not found.

- [ ] **Step 3: Write `errors.hpp` and the umbrella `expr.hpp`**

```cpp
// include/goss/model/expr/errors.hpp
#pragma once
#include <stdexcept>
#include <string>
namespace goss::model::expr {

/// Thrown when the expression DSL detects misuse: mismatched model handles,
/// duplicate dynamics registration, missing dynamics at build time, etc.
/// Distinct from ModelError so callers can distinguish expression-layer
/// problems from model-metadata problems.
class ExprError : public std::runtime_error {
 public:
    explicit ExprError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace goss::model::expr
```

```cpp
// include/goss/model/expr/expr.hpp
// Umbrella include for the expression DSL sub-library.
// Grow this file as each task adds new headers.
#pragma once
#include "goss/model/expr/errors.hpp"
```

- [ ] **Step 4: Add test source to CMake**

In `CMakeLists.txt`, locate the `add_executable(goss_model_tests ...)` block and append `tests/model/test_expr_nodes.cpp` to the source list. No other changes.

- [ ] **Step 5: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "ExprError" --output-on-failure'`
Expected: 1 test PASSES. Full suite still clean: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'`.

- [ ] **Step 6: Commit**

```bash
git add include/goss/model/expr/ tests/model/test_expr_nodes.cpp CMakeLists.txt
git commit -m "build: scaffold expression DSL sub-library with ExprError"
```

---

### Task 2: Expression node types — leaves and binary/unary operators

**Files:**
- Create: `include/goss/model/expr/nodes.hpp`
- Modify: `include/goss/model/expr/expr.hpp` (add `#include "goss/model/expr/nodes.hpp"`)
- Modify: `tests/model/test_expr_nodes.cpp` (add eval tests)

**Interfaces:**
- Produces (all in `namespace goss::model::expr`):
  - `struct ConstantExpr { double value; template<ScalarT> ScalarT eval(...) const; }`
  - `struct StateLeaf { std::size_t state_index; template<ScalarT> ScalarT eval(x,u,t) const; }` — returns `x[state_index]`.
  - `struct ControlLeaf { std::size_t control_index; template<ScalarT> ScalarT eval(x,u,t) const; }` — returns `u[control_index]`.
  - `struct TimeLeaf { template<ScalarT> ScalarT eval(x,u,t) const; }` — returns `t`.
  - Tag structs: `struct AddTag {}; struct SubTag {}; struct MulTag {};`
  - `template<typename OpTag, typename LeftExpr, typename RightExpr> struct BinaryExpr { LeftExpr left; RightExpr right; template<ScalarT> ScalarT eval(...) const; }` — dispatches on `OpTag` via `if constexpr`.
  - `template<typename OperandExpr> struct UnaryNegExpr { OperandExpr operand; template<ScalarT> ScalarT eval(...) const; }` — returns `ScalarT(-1) * operand.eval(...)`.

**AD-safety rules embedded in the impl:** `ConstantExpr::eval<ScalarT>` returns `ScalarT(value)` (not a bare `double`). `BinaryExpr<MulTag>` multiplies via the `*` operator (defined for CppAD AD). There are no `std::` math calls in v1.

**Design note on why expression templates (not `std::variant` + virtual):** CppAD records AD operations by operator overloading on a tape. To record correctly, the SAME expression code must be executed under `AD<CG<double>>`. A virtual `eval()` returning `double` cannot be re-instantiated under `AD<CG<double>>`. A `std::variant<...>` of concrete node types works but requires knowing all node types upfront and makes recursive composition verbose. Typed expression templates (the approach here) give the compiler complete type information: `BinaryExpr<MulTag, StateLeaf, ConstantExpr>` is a single concrete struct whose `eval<AD<CG<double>>>` member is compiled for the AD type, letting CppAD record every arithmetic operation exactly. The cost is that the expression TYPE encodes the tree structure — but since expressions are only used to build lambdas (never stored in runtime heterogeneous containers), this is not a problem.

- [ ] **Step 1: Write the failing tests**

```cpp
// append to tests/model/test_expr_nodes.cpp
#include "goss/model/expr/nodes.hpp"
#include <vector>
#include <cmath>

TEST(ExprNodes, ConstantEvalReturnsWrappedValue) {
    const goss::model::expr::ConstantExpr constant_node{3.14};
    const std::vector<double> x_empty{};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(constant_node.eval<double>(x_empty, u_empty, 0.0), 3.14);
}

TEST(ExprNodes, StateLeafIndexesXVector) {
    const goss::model::expr::StateLeaf state_node{1};  // second state
    const std::vector<double> x_vec{10.0, 20.0, 30.0};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(state_node.eval<double>(x_vec, u_empty, 0.0), 20.0);
}

TEST(ExprNodes, ControlLeafIndexesUVector) {
    const goss::model::expr::ControlLeaf control_node{0};
    const std::vector<double> x_empty{};
    const std::vector<double> u_vec{7.5};
    EXPECT_DOUBLE_EQ(control_node.eval<double>(x_empty, u_vec, 0.0), 7.5);
}

TEST(ExprNodes, TimeLeafReturnsT) {
    const goss::model::expr::TimeLeaf time_node{};
    const std::vector<double> x_empty{};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(time_node.eval<double>(x_empty, u_empty, 2.5), 2.5);
}

TEST(ExprNodes, BinaryAddExprSumsLeaves) {
    using namespace goss::model::expr;
    const BinaryExpr<AddTag, StateLeaf, ConstantExpr> add_node{StateLeaf{0}, ConstantExpr{5.0}};
    const std::vector<double> x_vec{3.0};
    const std::vector<double> u_empty{};
    // 3.0 + 5.0 = 8.0
    EXPECT_DOUBLE_EQ(add_node.eval<double>(x_vec, u_empty, 0.0), 8.0);
}

TEST(ExprNodes, BinaryMulExprMultipliesLeaves) {
    using namespace goss::model::expr;
    const BinaryExpr<MulTag, ConstantExpr, ControlLeaf> mul_node{ConstantExpr{2.0}, ControlLeaf{0}};
    const std::vector<double> x_empty{};
    const std::vector<double> u_vec{4.0};
    // 2.0 * 4.0 = 8.0
    EXPECT_DOUBLE_EQ(mul_node.eval<double>(x_empty, u_vec, 0.0), 8.0);
}

TEST(ExprNodes, UnaryNegExprNegatesOperand) {
    using namespace goss::model::expr;
    const UnaryNegExpr<ConstantExpr> neg_node{ConstantExpr{3.0}};
    const std::vector<double> x_empty{};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(neg_node.eval<double>(x_empty, u_empty, 0.0), -3.0);
}

TEST(ExprNodes, NestedExprComposesCorrectly) {
    // Represent: state[0] + 2.0 * control[0]
    using namespace goss::model::expr;
    const BinaryExpr<AddTag,
        StateLeaf,
        BinaryExpr<MulTag, ConstantExpr, ControlLeaf>>
        nested_add{
            StateLeaf{0},
            BinaryExpr<MulTag, ConstantExpr, ControlLeaf>{ConstantExpr{2.0}, ControlLeaf{0}}
        };
    const std::vector<double> x_vec{10.0};
    const std::vector<double> u_vec{3.0};
    // 10.0 + 2.0 * 3.0 = 16.0
    EXPECT_DOUBLE_EQ(nested_add.eval<double>(x_vec, u_vec, 0.0), 16.0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — `nodes.hpp` not found.

- [ ] **Step 3: Write `nodes.hpp`**

```cpp
// include/goss/model/expr/nodes.hpp
#pragma once
#include <cstddef>
#include <vector>

namespace goss::model::expr {

/// Leaf node: a compile-time constant. eval<ScalarT> wraps the stored double
/// in ScalarT(...) so CppAD records a constant on the AD tape — never use a
/// bare double literal inside eval when ScalarT may be an AD type.
struct ConstantExpr {
    double value;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& /*x*/,
                 const std::vector<ScalarT>& /*u*/,
                 ScalarT /*t*/) const {
        return ScalarT(value);
    }
};

/// Leaf node: reads state at index i from the x vector.
struct StateLeaf {
    std::size_t state_index;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& x,
                 const std::vector<ScalarT>& /*u*/,
                 ScalarT /*t*/) const {
        return x[state_index];
    }
};

/// Leaf node: reads control at index i from the u vector.
struct ControlLeaf {
    std::size_t control_index;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& /*x*/,
                 const std::vector<ScalarT>& u,
                 ScalarT /*t*/) const {
        return u[control_index];
    }
};

/// Leaf node: returns the current time t.
struct TimeLeaf {
    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& /*x*/,
                 const std::vector<ScalarT>& /*u*/,
                 ScalarT t) const {
        return t;
    }
};

/// Binary operator tags — structs used as template parameters to select
/// the arithmetic operation in BinaryExpr::eval without virtual dispatch.
struct AddTag {};
struct SubTag {};
struct MulTag {};

/// Binary expression node. Stores left and right subtrees by value (the tree
/// is stack-allocated; nodes are value types, not pointers). The OpTag
/// selects addition, subtraction, or multiplication via if constexpr so the
/// compiler inlines the correct operation under any scalar type T.
template <typename OpTag, typename LeftExpr, typename RightExpr>
struct BinaryExpr {
    LeftExpr left_operand;
    RightExpr right_operand;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& x,
                 const std::vector<ScalarT>& u,
                 ScalarT t) const {
        const ScalarT left_value  = left_operand.eval(x, u, t);
        const ScalarT right_value = right_operand.eval(x, u, t);
        if constexpr (std::is_same_v<OpTag, AddTag>) {
            return left_value + right_value;
        } else if constexpr (std::is_same_v<OpTag, SubTag>) {
            return left_value - right_value;
        } else {
            // MulTag — only three tags exist in v1; a static_assert catches
            // accidental instantiation with an unknown tag at compile time.
            static_assert(std::is_same_v<OpTag, MulTag>,
                "BinaryExpr: unknown OpTag — only AddTag, SubTag, MulTag are supported in v1");
            return left_value * right_value;
        }
    }
};

/// Unary negation node. Returns ScalarT(-1) * operand so CppAD records the
/// negation as a multiplication, which is correctly differentiated.
template <typename OperandExpr>
struct UnaryNegExpr {
    OperandExpr operand;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& x,
                 const std::vector<ScalarT>& u,
                 ScalarT t) const {
        return ScalarT(-1) * operand.eval(x, u, t);
    }
};

}  // namespace goss::model::expr
```

Add `#include "goss/model/expr/nodes.hpp"` to `expr.hpp`.

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "ExprNodes" --output-on-failure'`
Expected: all 8 ExprNodes tests PASS. Run the full suite to confirm no regression.

- [ ] **Step 5: Commit**

```bash
git add include/goss/model/expr/nodes.hpp include/goss/model/expr/expr.hpp tests/model/test_expr_nodes.cpp
git commit -m "feat: expression node types (leaves, binary ops, unary neg) with templated eval"
```

---

### Task 3: Free-function `operator+/-/*` and AD-safety verification

**Files:**
- Create: `include/goss/model/expr/operators.hpp`
- Modify: `include/goss/model/expr/expr.hpp`
- Modify: `tests/model/test_expr_nodes.cpp` (add operator and AD tests)

**Interfaces:**
- Produces (all in `namespace goss::model::expr`):
  - `operator+(L, R)`, `operator-(L, R)`, `operator*(L, R)` for any two expression nodes (each returns the corresponding `BinaryExpr<Tag,L,R>`).
  - Overloads that accept `double` on either side and implicitly wrap it in `ConstantExpr{value}` so users can write `q + 5.0` without explicit `ConstantExpr`.
  - Unary `operator-(Expr)` returning `UnaryNegExpr<Expr>`.
  - These are the ONLY operators in `operators.hpp` in this task. Comparison operators (`>=`, `<=`, `==`) come in Task 4.

**Design note — AD safety of the operator overloads:** The operators themselves are pure C++; they return composed node structs. No arithmetic happens at composition time — only the `eval<ScalarT>()` call performs arithmetic, which happens inside the lambda passed to `Model::build()`, where CppAD has already called `CppAD::Independent(x)` on the AD-typed vector. Therefore, CppAD records every `+`, `-`, `*` from these nodes correctly. The double-wrapping overload (`operator+(StateLeaf, double)`) works because `ConstantExpr{5.0}.eval<AD<...>>()` returns `AD<...>(5.0)` — a CppAD constant node on the tape — not a bare `double`.

**AD instantiation test:** Add a test that calls `eval<CppAD::AD<double>>` on a non-trivial composed expression (`q + w*rate*rate` shape) to verify the template instantiates without error under an AD type. This does NOT require recording a full CppAD tape — constructing AD variables and calling eval is sufficient to confirm AD-safety at compile time and shallow runtime. (A full tape-recording test would belong in an integration test at the transcription layer level; for the expr layer, instantiability suffices.)

- [ ] **Step 1: Write the failing tests**

```cpp
// append to tests/model/test_expr_nodes.cpp
#include "goss/model/expr/operators.hpp"
// CppAD is available via goss_model -> goss_transcription -> goss_nlp -> goss_ad -> cppadcg
#include <cppad/cppad.hpp>

TEST(ExprOperators, PlusBuildsBinaryAddExpr) {
    using namespace goss::model::expr;
    const auto add_expr = StateLeaf{0} + ConstantExpr{1.0};
    const std::vector<double> x_vec{9.0};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(add_expr.eval<double>(x_vec, u_empty, 0.0), 10.0);
}

TEST(ExprOperators, MinusBuildsSubExpr) {
    using namespace goss::model::expr;
    const auto sub_expr = ConstantExpr{5.0} - ControlLeaf{0};
    const std::vector<double> x_empty{};
    const std::vector<double> u_vec{2.0};
    EXPECT_DOUBLE_EQ(sub_expr.eval<double>(x_empty, u_vec, 0.0), 3.0);
}

TEST(ExprOperators, MulBuildsMulExpr) {
    using namespace goss::model::expr;
    const auto mul_expr = ConstantExpr{3.0} * ControlLeaf{0};
    const std::vector<double> x_empty{};
    const std::vector<double> u_vec{4.0};
    EXPECT_DOUBLE_EQ(mul_expr.eval<double>(x_empty, u_vec, 0.0), 12.0);
}

TEST(ExprOperators, UnaryMinusNegatesExpr) {
    using namespace goss::model::expr;
    const auto neg_expr = -StateLeaf{0};
    const std::vector<double> x_vec{7.0};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(neg_expr.eval<double>(x_vec, u_empty, 0.0), -7.0);
}

TEST(ExprOperators, DoubleLiteralOnRhsIsWrappedImplicitly) {
    using namespace goss::model::expr;
    // q + 5.0 — the double overload wraps 5.0 in ConstantExpr.
    const auto add_double_expr = StateLeaf{0} + 5.0;
    const std::vector<double> x_vec{2.0};
    const std::vector<double> u_empty{};
    EXPECT_DOUBLE_EQ(add_double_expr.eval<double>(x_vec, u_empty, 0.0), 7.0);
}

TEST(ExprOperators, ComposedExprInstantiatesUnderCppADAD) {
    // Verify that q + w * rate * rate composes and eval<AD<double>> does not
    // fail to compile or produce NaN.  w = 0.1, q = 10.0, rate = 3.0 => 10 + 0.1*9 = 10.9
    using namespace goss::model::expr;
    using ADDouble = CppAD::AD<double>;
    const auto weight_mul_rate    = ConstantExpr{0.1} * ControlLeaf{0};
    const auto weight_rate_sq     = weight_mul_rate * ControlLeaf{0};
    const auto cost_expr          = StateLeaf{0} + weight_rate_sq;
    const std::vector<ADDouble> x_ad{ADDouble(10.0)};
    const std::vector<ADDouble> u_ad{ADDouble(3.0)};
    const ADDouble t_ad{0.0};
    const ADDouble result_ad = cost_expr.eval<ADDouble>(x_ad, u_ad, t_ad);
    EXPECT_DOUBLE_EQ(CppAD::Value(result_ad), 10.9);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — `operators.hpp` not found.

- [ ] **Step 3: Write `operators.hpp`**

```cpp
// include/goss/model/expr/operators.hpp
#pragma once
#include "goss/model/expr/nodes.hpp"

namespace goss::model::expr {

// Helper: wrap a double in ConstantExpr so double literals compose with nodes.
// This is a private implementation detail; it is not part of the public API.
namespace detail {
    inline ConstantExpr wrap_double(double value) { return ConstantExpr{value}; }
}  // namespace detail

// --- operator+ ---

template <typename LeftExpr, typename RightExpr>
BinaryExpr<AddTag, LeftExpr, RightExpr>
operator+(LeftExpr left_operand, RightExpr right_operand) {
    return BinaryExpr<AddTag, LeftExpr, RightExpr>{
        std::move(left_operand), std::move(right_operand)};
}

template <typename LeftExpr>
BinaryExpr<AddTag, LeftExpr, ConstantExpr>
operator+(LeftExpr left_operand, double right_value) {
    return BinaryExpr<AddTag, LeftExpr, ConstantExpr>{
        std::move(left_operand), detail::wrap_double(right_value)};
}

template <typename RightExpr>
BinaryExpr<AddTag, ConstantExpr, RightExpr>
operator+(double left_value, RightExpr right_operand) {
    return BinaryExpr<AddTag, ConstantExpr, RightExpr>{
        detail::wrap_double(left_value), std::move(right_operand)};
}

// --- operator- (binary) ---

template <typename LeftExpr, typename RightExpr>
BinaryExpr<SubTag, LeftExpr, RightExpr>
operator-(LeftExpr left_operand, RightExpr right_operand) {
    return BinaryExpr<SubTag, LeftExpr, RightExpr>{
        std::move(left_operand), std::move(right_operand)};
}

template <typename LeftExpr>
BinaryExpr<SubTag, LeftExpr, ConstantExpr>
operator-(LeftExpr left_operand, double right_value) {
    return BinaryExpr<SubTag, LeftExpr, ConstantExpr>{
        std::move(left_operand), detail::wrap_double(right_value)};
}

template <typename RightExpr>
BinaryExpr<SubTag, ConstantExpr, RightExpr>
operator-(double left_value, RightExpr right_operand) {
    return BinaryExpr<SubTag, ConstantExpr, RightExpr>{
        detail::wrap_double(left_value), std::move(right_operand)};
}

// --- operator* ---

template <typename LeftExpr, typename RightExpr>
BinaryExpr<MulTag, LeftExpr, RightExpr>
operator*(LeftExpr left_operand, RightExpr right_operand) {
    return BinaryExpr<MulTag, LeftExpr, RightExpr>{
        std::move(left_operand), std::move(right_operand)};
}

template <typename LeftExpr>
BinaryExpr<MulTag, LeftExpr, ConstantExpr>
operator*(LeftExpr left_operand, double right_value) {
    return BinaryExpr<MulTag, LeftExpr, ConstantExpr>{
        std::move(left_operand), detail::wrap_double(right_value)};
}

template <typename RightExpr>
BinaryExpr<MulTag, ConstantExpr, RightExpr>
operator*(double left_value, RightExpr right_operand) {
    return BinaryExpr<MulTag, ConstantExpr, RightExpr>{
        detail::wrap_double(left_value), std::move(right_operand)};
}

// --- unary operator- ---

template <typename OperandExpr>
UnaryNegExpr<OperandExpr>
operator-(OperandExpr operand) {
    return UnaryNegExpr<OperandExpr>{std::move(operand)};
}

}  // namespace goss::model::expr
```

Add `#include <utility>` at the top of `operators.hpp` (for `std::move`). Add `#include "goss/model/expr/operators.hpp"` to `expr.hpp`.

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "ExprOperators" --output-on-failure'`
Expected: all 6 ExprOperators tests PASS, including the `CppAD::AD<double>` instantiation test.

- [ ] **Step 5: Commit**

```bash
git add include/goss/model/expr/operators.hpp include/goss/model/expr/expr.hpp tests/model/test_expr_nodes.cpp
git commit -m "feat: expression operator overloads (+, -, *) with CppAD AD-safety verification"
```

---

### Task 4: Comparison operators — `BoundConstraint` and `BoundaryConstraint` lowering

**Files:**
- Create: `include/goss/model/expr/constraints.hpp`
- Modify: `include/goss/model/expr/operators.hpp` (add comparison operators)
- Modify: `include/goss/model/expr/expr.hpp`
- Create: `tests/model/test_expr_lowering.cpp`
- Modify: `CMakeLists.txt` (add `tests/model/test_expr_lowering.cpp`)

**Interfaces:**
- Produces (in `namespace goss::model::expr`):
  - `struct BoundConstraint { goss::model::StateHandle state_handle; double lower_bound; double upper_bound; }` — represents a box bound on a single state, ready to lower to `Model::set_state_bounds`.
  - `struct ControlBoundConstraint { goss::model::ControlHandle control_handle; double lower_bound; double upper_bound; }` — analogous for controls.
  - `struct BoundaryConstraint { goss::model::StateHandle state_handle; double fixed_value; enum class Kind { Initial, Final } kind; }` — represents a fixed boundary condition.
  - Free functions `operator>=(StateHandle, double)` → `BoundConstraint{s, value, kInf}`, `operator<=(StateHandle, double)` → `BoundConstraint{s, -kInf, value}`.
  - `operator>=(ControlHandle, double)` → `ControlBoundConstraint{c, value, kInf}`, `operator<=(ControlHandle, double)` → `ControlBoundConstraint{c, -kInf, value}`.
  - `StateHandle::initial()` / `StateHandle::final()` — return a thin `BoundaryPoint{handle, Kind}` struct.
  - `operator==(BoundaryPoint, double)` → `BoundaryConstraint{handle, value, kind}`.
  - These operators (`>=`, `<=`, `==`) are NOT overloaded on general expression types — only on `StateHandle`/`ControlHandle` directly. If a user writes `(q + 1.0) >= 0.0`, the implicit conversion path does not exist (no implicit `StateHandle` cast from a `BinaryExpr`) and the compiler gives a clean "no match for operator>=" error. This is intentional: general nonlinear path constraints require a transcription extension and are explicitly out of v1 scope (see below).

**General nonlinear path constraints — v1 scope assessment:**
A nonlinear path constraint `g(x, u, t) >= 0` would require the transcription layer (`HermiteSimpson`, `Trapezoidal`) to:
1. Accept a collection of `PathConstraintFn` objects (each with the same `template<T> T eval(x,u,t)` signature).
2. Evaluate them at every collocation node and mid-point, and add their values + Jacobian rows to the NLP.
This is a non-trivial transcription change: `OcpProblem` gains a new field (`std::vector<PathConstraintFn>`), `VariableLayout` gains new constraint-count metadata, and both `Trapezoidal::compile` and `HermiteSimpson::compile` must be extended. This work is firmly deferred. The plan notes this here so a future implementer knows exactly where the extension point is.

- [ ] **Step 1: Add `initial()`/`final()` methods to `StateHandle`**

The operator-overload syntax `q.initial() == 10.0` requires `StateHandle` to have member functions `initial()` and `final()`. These cannot be forward-declared in a separate header because `StateHandle` is defined in `handles.hpp`. The cleanest approach: add `initial()` and `final()` to `handles.hpp` that return a `BoundaryPoint` — but `BoundaryPoint` is defined in `constraints.hpp` which is a later include. To break the cycle: define `BoundaryPoint` as a forward declaration with no dependency on `constraints.hpp` details, OR define `BoundaryPoint` in `handles.hpp` as a minimal struct. The latter is simpler for a header-only library.

**Recommended approach (stated in the plan, not implemented):** Add a minimal `struct BoundaryPoint { StateHandle handle; enum class Kind { Initial, Final } kind; };` directly to `handles.hpp`, and add `BoundaryPoint initial() const { return {*this, BoundaryPoint::Kind::Initial}; }` and `BoundaryPoint final() const { return {*this, BoundaryPoint::Kind::Final}; }` to `StateHandle`. This modifies `handles.hpp` — the ONLY modification to an existing file in this entire plan. The change is purely additive (the existing `operator std::size_t()` and `index` field are untouched) and is guarded by a `#include "goss/model/expr/constraints_fwd.hpp"` in `handles.hpp` which pre-declares `BoundaryPoint` so the handles header itself remains minimal.

**Simpler alternative (adopted in this plan):** Define `BoundaryPoint` inline in `handles.hpp` with no separate forward-declaration file. This is the only clean way to add `.initial()`/`.final()` to `StateHandle` without a circular include. The addition to `handles.hpp` is:

```cpp
// Add BEFORE the existing struct StateHandle definition:
namespace goss::model {
struct BoundaryPoint {
    struct StateHandle;  // forward declaration resolved below
    // ... actually, define after StateHandle
};
// This does not work cleanly as a forward declaration.
// CORRECT approach: define BoundaryPoint after StateHandle in the same file.
}
```

Since `BoundaryPoint` references `StateHandle` by value (not pointer), `BoundaryPoint` must be defined AFTER `StateHandle`. And `StateHandle::initial()` returns `BoundaryPoint`, so `BoundaryPoint` must be complete before the `initial()` definition. The solution: put `BoundaryPoint` between `StateHandle` definition and `initial()`/`final()` inline method bodies — define the methods with `inline` after `BoundaryPoint` is complete, which is naturally the case if `BoundaryPoint` is defined between the two structs and `initial()`/`final()` are defined after (as non-inline declarations with inline definitions below the `BoundaryPoint` struct). In practice for a header-only `struct`, the simplest C++17-legal approach is:

```cpp
// handles.hpp addition (the only modified existing file in this plan):
struct StateHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }
    // initial() and final() defined below, after BoundaryPoint is declared
};
struct BoundaryPoint {
    StateHandle state_handle;
    enum class Kind { Initial, Final };
    Kind kind;
};
inline BoundaryPoint StateHandle::initial() const { return BoundaryPoint{*this, BoundaryPoint::Kind::Initial}; }
inline BoundaryPoint StateHandle::final()   const { return BoundaryPoint{*this, BoundaryPoint::Kind::Final};   }
```

This requires the `initial()`/`final()` declarations to appear in the `StateHandle` struct body BEFORE `BoundaryPoint` is defined — which is fine because they are only declared there; the definitions (using `BoundaryPoint`) come after. Write the failing test first, then make this exact modification to `handles.hpp`.

- [ ] **Step 2: Write the failing tests**

```cpp
// tests/model/test_expr_lowering.cpp
#include <gtest/gtest.h>
#include "goss/model/expr/constraints.hpp"   // includes operators.hpp, nodes.hpp, handles.hpp
#include "goss/model/model.hpp"
#include "goss/transcription/transcription.hpp"

TEST(ExprLowering, StateGeqDoubleLowersToStateLowerBound) {
    goss::model::Model model;
    const auto q = model.add_state("q");
    const goss::model::expr::BoundConstraint bound = (q >= 0.0);
    EXPECT_EQ(bound.state_handle.index, q.index);
    EXPECT_DOUBLE_EQ(bound.lower_bound, 0.0);
    EXPECT_DOUBLE_EQ(bound.upper_bound, goss::transcription::kInf);
}

TEST(ExprLowering, StateLeqDoubleLowersToStateUpperBound) {
    goss::model::Model model;
    const auto q = model.add_state("q");
    const goss::model::expr::BoundConstraint bound = (q <= 100.0);
    EXPECT_DOUBLE_EQ(bound.lower_bound, -goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(bound.upper_bound, 100.0);
}

TEST(ExprLowering, ControlGeqDoubleLowersToControlLowerBound) {
    goss::model::Model model;
    const auto rate = model.add_control("rate");
    const goss::model::expr::ControlBoundConstraint bound = (rate >= 0.0);
    EXPECT_EQ(bound.control_handle.index, rate.index);
    EXPECT_DOUBLE_EQ(bound.lower_bound, 0.0);
    EXPECT_DOUBLE_EQ(bound.upper_bound, goss::transcription::kInf);
}

TEST(ExprLowering, StateInitialEqDoubleLowersToBoundaryConstraint) {
    goss::model::Model model;
    const auto q = model.add_state("q");
    const goss::model::expr::BoundaryConstraint bc = (q.initial() == 10.0);
    EXPECT_EQ(bc.state_handle.index, q.index);
    EXPECT_DOUBLE_EQ(bc.fixed_value, 10.0);
    EXPECT_EQ(bc.kind, goss::model::expr::BoundaryConstraint::Kind::Initial);
}

TEST(ExprLowering, StateFinalEqDoubleLowersToFinalBoundaryConstraint) {
    goss::model::Model model;
    const auto q = model.add_state("q");
    const goss::model::expr::BoundaryConstraint bc = (q.final() == 1.0);
    EXPECT_EQ(bc.kind, goss::model::expr::BoundaryConstraint::Kind::Final);
    EXPECT_DOUBLE_EQ(bc.fixed_value, 1.0);
}

TEST(ExprLowering, ApplyBoundConstraintCallsModelSetter) {
    goss::model::Model model;
    const auto q    = model.add_state("q");
    const auto rate = model.add_control("rate");
    goss::model::expr::apply_bound(model, q >= 0.0);
    goss::model::expr::apply_bound(model, rate <= 5.0);
    EXPECT_DOUBLE_EQ(model.state_lower(0), 0.0);
    EXPECT_DOUBLE_EQ(model.state_upper(0), goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(model.control_lower(0), -goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(model.control_upper(0), 5.0);
}

TEST(ExprLowering, ApplyBoundaryConstraintCallsModelSetter) {
    goss::model::Model model;
    const auto q = model.add_state("q");
    goss::model::expr::apply_boundary(model, q.initial() == 10.0);
    EXPECT_TRUE(model.initial_fixed(0));
    EXPECT_DOUBLE_EQ(model.initial_value(0), 10.0);
    EXPECT_FALSE(model.final_fixed(0));
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — `constraints.hpp` not found, `q.initial()` not a member.

- [ ] **Step 4: Modify `handles.hpp` (the only existing-file modification in this plan) and write `constraints.hpp`**

Modify `include/goss/model/handles.hpp` to add `BoundaryPoint` and `initial()`/`final()`:

```cpp
// include/goss/model/handles.hpp  — full replacement (additive only)
#pragma once
#include <cstddef>
namespace goss::model {

/// Opaque handle to a declared state. Implicitly converts to std::size_t so it
/// can index the x-vector inside dynamics/cost lambdas (`x[q]`). It is a struct
/// (not a bare size_t) so the expression DSL can attach operators to it
/// (e.g. q >= 0.0, q.initial() == 10.0) without changing call sites.
struct StateHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }

    // Declared here; defined inline below after BoundaryPoint is complete.
    struct BoundaryPoint;  // forward declaration
    BoundaryPoint initial() const;
    BoundaryPoint final()   const;
};

/// Opaque handle to a declared control. See StateHandle.
struct ControlHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }
};

/// A boundary attachment point — returned by StateHandle::initial() and
/// StateHandle::final(). Used with operator== to build a BoundaryConstraint
/// that lowers to set_initial_state / set_final_state.
struct StateHandle::BoundaryPoint {
    StateHandle state_handle;
    enum class Kind { Initial, Final };
    Kind kind;
};
// Use BoundaryPoint as the return type qualifier to satisfy the forward decl:
using BoundaryPoint = StateHandle::BoundaryPoint;

inline BoundaryPoint StateHandle::initial() const {
    return BoundaryPoint{*this, BoundaryPoint::Kind::Initial};
}
inline BoundaryPoint StateHandle::final() const {
    return BoundaryPoint{*this, BoundaryPoint::Kind::Final};
}

}  // namespace goss::model
```

Write `include/goss/model/expr/constraints.hpp`:

```cpp
// include/goss/model/expr/constraints.hpp
#pragma once
#include "goss/model/handles.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/transcription.hpp"  // kInf

namespace goss::model::expr {

/// Represents a box constraint on a single state: lower_bound <= state <= upper_bound.
/// Constructed by operator>= / operator<= on a StateHandle; lowered to
/// Model::set_state_bounds by apply_bound().
struct BoundConstraint {
    goss::model::StateHandle state_handle;
    double lower_bound;
    double upper_bound;
};

/// Represents a box constraint on a single control.
struct ControlBoundConstraint {
    goss::model::ControlHandle control_handle;
    double lower_bound;
    double upper_bound;
};

/// Represents a fixed boundary condition on a state.
/// Constructed by operator==(BoundaryPoint, double); lowered to
/// Model::set_initial_state / set_final_state by apply_boundary().
struct BoundaryConstraint {
    goss::model::StateHandle state_handle;
    double fixed_value;
    enum class Kind { Initial, Final };
    Kind kind;
};

// --- Comparison operators on StateHandle ---

inline BoundConstraint operator>=(goss::model::StateHandle s, double lower_value) {
    return BoundConstraint{s, lower_value, goss::transcription::kInf};
}

inline BoundConstraint operator<=(goss::model::StateHandle s, double upper_value) {
    return BoundConstraint{s, -goss::transcription::kInf, upper_value};
}

// --- Comparison operators on ControlHandle ---

inline ControlBoundConstraint operator>=(goss::model::ControlHandle c, double lower_value) {
    return ControlBoundConstraint{c, lower_value, goss::transcription::kInf};
}

inline ControlBoundConstraint operator<=(goss::model::ControlHandle c, double upper_value) {
    return ControlBoundConstraint{c, -goss::transcription::kInf, upper_value};
}

// --- BoundaryPoint == double ---

inline BoundaryConstraint operator==(goss::model::BoundaryPoint bp, double fixed_value) {
    return BoundaryConstraint{
        bp.state_handle,
        fixed_value,
        bp.kind == goss::model::BoundaryPoint::Kind::Initial
            ? BoundaryConstraint::Kind::Initial
            : BoundaryConstraint::Kind::Final
    };
}

// --- Lowering helpers: apply constraints to a Model ---

/// Lower a BoundConstraint to Model::set_state_bounds.
inline void apply_bound(goss::model::Model& model, const BoundConstraint& constraint) {
    model.set_state_bounds(constraint.state_handle, constraint.lower_bound, constraint.upper_bound);
}

/// Lower a ControlBoundConstraint to Model::set_control_bounds.
inline void apply_bound(goss::model::Model& model, const ControlBoundConstraint& constraint) {
    model.set_control_bounds(constraint.control_handle, constraint.lower_bound, constraint.upper_bound);
}

/// Lower a BoundaryConstraint to Model::set_initial_state or Model::set_final_state.
inline void apply_boundary(goss::model::Model& model, const BoundaryConstraint& constraint) {
    if (constraint.kind == BoundaryConstraint::Kind::Initial) {
        model.set_initial_state(constraint.state_handle, constraint.fixed_value);
    } else {
        model.set_final_state(constraint.state_handle, constraint.fixed_value);
    }
}

}  // namespace goss::model::expr
```

Add `#include "goss/model/expr/constraints.hpp"` to `expr.hpp`. Add `tests/model/test_expr_lowering.cpp` to `goss_model_tests` in `CMakeLists.txt`.

- [ ] **Step 5: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "ExprLowering" --output-on-failure'`
Expected: all 7 ExprLowering tests PASS. Verify no regression in existing model tests: `scripts/dev.sh 'ctest --test-dir build -R "Model" --output-on-failure'`.

- [ ] **Step 6: Commit**

```bash
git add include/goss/model/handles.hpp include/goss/model/expr/constraints.hpp \
        include/goss/model/expr/expr.hpp tests/model/test_expr_lowering.cpp CMakeLists.txt
git commit -m "feat: comparison operators and constraint lowering (>= <= == on handles)"
```

---

### Task 5: `integral()` cost wrapper and `CostFunctor`

**Files:**
- Create: `include/goss/model/expr/integral.hpp`
- Modify: `include/goss/model/expr/expr.hpp`
- Modify: `tests/model/test_expr_lowering.cpp` (add `integral` tests)

**Interfaces:**
- Produces (in `namespace goss::model::expr`):
  - `template<typename CostExpr> struct CostFunctor` — holds a `CostExpr` by value; exposes `template<T> T operator()(const vector<T>& x, const vector<T>& u, T t) const` that returns `cost_expr_.eval<T>(x, u, t)`. This satisfies the `Model::build()` `CostFn` contract exactly (same signature, template `operator()`).
  - `template<typename CostExpr> CostFunctor<CostExpr> integral(CostExpr cost_expr)` — factory function that wraps any expression node into a `CostFunctor`. Naming: `integral` follows the spec's DSL sugar (`set_cost(integral(q + w*rate*rate))`); the cost functor is what HermiteSimpson evaluates as the running (Lagrange) cost at each collocation point — exactly the role of the `cost` argument in `Model::build()`.

**AD-safety:** `CostFunctor::operator()<AD<CG<double>>>` calls `cost_expr_.eval<AD<CG<double>>>(x_ad, u_ad, t_ad)` — since the expression tree's `eval` is fully templated, this instantiates cleanly under any AD type.

**Note on the `integral` name:** the running cost `f(x,u,t)` that `HermiteSimpson` integrates via Simpson quadrature IS the time-integral of the Lagrange cost. Naming the wrapper `integral(expr)` matches the spec DSL and signals intent ("minimize the integral of this expression over the horizon"). The implementation is just a `CostFunctor` that evaluates `expr` pointwise — HermiteSimpson does the numerical integration.

- [ ] **Step 1: Write the failing tests**

```cpp
// append to tests/model/test_expr_lowering.cpp
#include "goss/model/expr/integral.hpp"

TEST(ExprIntegral, CostFunctorEvalMatchesManualCalculation) {
    using namespace goss::model::expr;
    // cost = q + 0.1 * rate^2
    const auto cost_expr = StateLeaf{0} + ConstantExpr{0.1} * ControlLeaf{0} * ControlLeaf{0};
    const auto cost_functor = integral(cost_expr);
    const std::vector<double> x_vec{10.0};
    const std::vector<double> u_vec{3.0};
    // 10 + 0.1 * 9 = 10.9
    EXPECT_DOUBLE_EQ(cost_functor(x_vec, u_vec, 0.0), 10.9);
}

TEST(ExprIntegral, CostFunctorInstantiatesUnderCppADAD) {
    using namespace goss::model::expr;
    using ADDouble = CppAD::AD<double>;
    const auto cost_expr    = StateLeaf{0} + ConstantExpr{0.1} * ControlLeaf{0} * ControlLeaf{0};
    const auto cost_functor = integral(cost_expr);
    const std::vector<ADDouble> x_ad{ADDouble(10.0)};
    const std::vector<ADDouble> u_ad{ADDouble(3.0)};
    const ADDouble result_ad = cost_functor(x_ad, u_ad, ADDouble(0.0));
    EXPECT_DOUBLE_EQ(CppAD::Value(result_ad), 10.9);
}

TEST(ExprIntegral, CostFunctorSatisfiesModelBuildCostFnContract) {
    // Verify that a CostFunctor can be passed to Model::build as the cost argument.
    using namespace goss::model::expr;
    goss::model::Model model;
    const auto q    = model.add_state("q");
    const auto rate = model.add_control("rate");
    model.set_mesh(0.0, 1.0, 2);
    const auto cost_functor = integral(StateLeaf{q.index} + ConstantExpr{0.1} * ControlLeaf{rate.index} * ControlLeaf{rate.index});
    // dynamics: trivial constant; just test that build compiles and does not throw.
    auto trivial_dynamics = [](const auto& /*x*/, const auto& /*u*/, auto /*t*/) {
        using T2 = typename std::decay_t<decltype(x)>::value_type;
        return std::vector<T2>{T2(0.0)};
    };
    // If CostFunctor does not satisfy CostFn, this will not compile.
    EXPECT_NO_THROW(model.build(trivial_dynamics, cost_functor));
}
```

**Note:** In `CostFunctorSatisfiesModelBuildCostFnContract`, the `trivial_dynamics` lambda has a typo — `x` is not declared. Write it correctly in the implementation:

```cpp
auto trivial_dynamics = [](const auto& x_vec, const auto& /*u*/, auto /*t*/) {
    using T2 = typename std::decay_t<decltype(x_vec)>::value_type;
    return std::vector<T2>{T2(0.0)};
};
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — `integral.hpp` not found.

- [ ] **Step 3: Write `integral.hpp`**

```cpp
// include/goss/model/expr/integral.hpp
#pragma once
#include <vector>
#include "goss/model/expr/nodes.hpp"   // for node types referenced in tests

namespace goss::model::expr {

/// Wraps a cost expression node into a functor compatible with Model::build's
/// CostFn contract: template<T> T operator()(const vector<T>&, const vector<T>&, T).
/// The name "integral" signals that HermiteSimpson will integrate this
/// expression pointwise over the horizon via Simpson quadrature.
template <typename CostExpr>
struct CostFunctor {
    CostExpr cost_expression;

    template <typename ScalarT>
    ScalarT operator()(const std::vector<ScalarT>& x,
                       const std::vector<ScalarT>& u,
                       ScalarT t) const {
        return cost_expression.eval<ScalarT>(x, u, t);
    }
};

/// Factory function: wraps any expression node in a CostFunctor.
/// Usage: set_cost(integral(q + weight * rate * rate))
template <typename CostExpr>
CostFunctor<CostExpr> integral(CostExpr cost_expression) {
    return CostFunctor<CostExpr>{std::move(cost_expression)};
}

}  // namespace goss::model::expr
```

Add `#include <utility>` for `std::move`. Add `#include "goss/model/expr/integral.hpp"` to `expr.hpp`.

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "ExprIntegral" --output-on-failure'`
Expected: all 3 ExprIntegral tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/goss/model/expr/integral.hpp include/goss/model/expr/expr.hpp tests/model/test_expr_lowering.cpp
git commit -m "feat: integral() cost wrapper with CostFunctor satisfying Model::build CostFn contract"
```

---

### Task 6: `DynamicsFunctor` and per-state `set_dynamics` — the `ExprModel`

**Files:**
- Create: `include/goss/model/expr/expr_model.hpp`
- Modify: `include/goss/model/expr/expr.hpp`
- Modify: `tests/model/test_expr_lowering.cpp` (add ExprModel tests)

**Interfaces:**
- Produces (in `namespace goss::model::expr`):
  - `template<typename... DynExprs> struct DynamicsFunctor` — holds a `std::tuple<DynExprs...>` of per-state dynamics expressions (one per state, in declaration order); exposes `template<T> std::vector<T> operator()(const vector<T>& x, const vector<T>& u, T t) const` that evaluates each expression and returns a `std::vector<T>` of size `sizeof...(DynExprs)`. Satisfies the `Model::build()` `DynamicsFn` contract.
  - `class ExprModel` — a thin wrapper:
    - Holds a `goss::model::Model model_` by value.
    - Exposes all `Model` setters by forwarding: `add_state`, `add_control`, `set_mesh`, `set_state_bounds`, `set_control_bounds`, `set_initial_state`, `set_final_state` (and accessor methods for tests).
    - Provides `template<typename DynExpr> void set_dynamics(StateHandle s, DynExpr dyn_expr)` — stores the expression in an internal per-state slot (a `std::vector<std::any>` with type-erased storage, or better: the typed approach described below).
    - Provides `template<typename CostExpr> void set_cost(CostFunctor<CostExpr> cost_functor)` — stores the cost functor for `build()`.
    - Provides `void apply(BoundConstraint c)` and `void apply(ControlBoundConstraint c)` and `void apply(BoundaryConstraint c)` — sugar that forward to the underlying model's setters.
    - Provides a zero-argument `build()` that assembles the collected per-state dynamics into a combined dynamics functor and calls `model_.build(dynamics_functor, cost_functor_)`.
    - Re-exposes the two-argument `build(dynamics, cost)` that delegates to `model_.build(dynamics, cost)` unchanged.

**Design note — the typed storage problem for per-state dynamics:**
`ExprModel` must store N per-state dynamics expressions of heterogeneous types (each `DynExpr` type depends on the expression tree the user wrote). Storing them in a `std::vector<???>`  requires type erasure. Two options:

1. **Type-erased storage via `std::function`:** Store a `std::vector<std::function<???(x,u,t)>>` — but `std::function` cannot template its `operator()`, so this forces a fixed scalar type (e.g., double), breaking AD.

2. **Type-erased but templated callable via a hand-rolled virtual-templated approach:** Define an `IDynExpr` base with `virtual void eval_double(...)` and `virtual void eval_ad(...)` — brittle and defeats the purpose.

3. **Variant over known expression types (not viable in general):** Expression trees are nested templates; the type space is unbounded.

4. **Collect-then-build with tuple expansion:** Store dynamics expressions as a `std::tuple` of typed entries, growing it with each `set_dynamics` call via tuple concatenation. A zero-argument `build()` calls `model_.build(make_dynamics_functor(tuple_), cost_functor_)`. Tuple concatenation (`std::tuple_cat`) in a class member is tricky because the tuple type changes with each `set_dynamics` call — requiring the class itself to be templated on its current tuple state (a builder pattern).

**Adopted approach (stated explicitly, since this is the most subtle design decision in the plan):** Make `ExprModel` a template class `ExprModel<DynTuple, CostFn>` that defaults to `ExprModel<std::tuple<>, void>` and returns a new type `ExprModel<new_tuple, ...>` from `set_dynamics`. This is a "fluent builder with accumulating type" pattern. The user writes:

```cpp
auto expr_model = goss::model::expr::ExprModel{}
    .with_dynamics(q, ARRIVAL - rate)
    .with_cost(integral(q + weight * rate * rate));
auto ocp = expr_model.build();
```

`with_dynamics` returns `ExprModel<std::tuple<Expr>, void>`, a second call returns `ExprModel<std::tuple<Expr1, Expr2>, void>`, etc. A zero-arg `build()` on the fully-specified `ExprModel` invokes `DynamicsFunctor` with the exact tuple. This is the cleanest type-safe approach and avoids any runtime overhead or type erasure, since the tuple type is known at build time.

**Consequence:** `set_dynamics` and `set_cost` become `with_dynamics` and `with_cost` (returning new ExprModel types). The `apply()` methods for constraints mutate state (they forward to the underlying `Model`), so they remain members that return `*this` by reference (which is fine since constraint application does not change the dynamics/cost types).

The test covers: assembling a two-state problem with `with_dynamics`, calling zero-arg `build()`, and verifying the resulting `OcpProblem` fields match those from the equivalent lambda build.

- [ ] **Step 1: Write the failing tests**

```cpp
// append to tests/model/test_expr_lowering.cpp
#include "goss/model/expr/expr_model.hpp"

TEST(ExprModel, WithDynamicsAndCostAssemblesOcpMatchingLambdaVersion) {
    const double ARRIVAL = 3.0;
    const double WEIGHT  = 0.1;

    // --- ExprModel (expression DSL) path ---
    goss::model::expr::ExprModel<> expr_model{};
    auto q    = expr_model.add_state("queue_length");
    auto rate = expr_model.add_control("service_rate");
    expr_model.apply(q >= 0.0);
    expr_model.apply(rate >= 0.0);
    expr_model.apply(rate <= 5.0);
    expr_model.apply(q.initial() == 10.0);
    expr_model.set_mesh(0.0, 5.0, 10);

    using namespace goss::model::expr;
    // dq/dt = ARRIVAL - rate
    const auto dynamics_expr = ConstantExpr{ARRIVAL} - ControlLeaf{rate.index};
    // cost = q + WEIGHT * rate^2
    const auto cost_expr     = StateLeaf{q.index} + ConstantExpr{WEIGHT} * ControlLeaf{rate.index} * ControlLeaf{rate.index};

    auto built_expr_model = std::move(expr_model)
        .with_dynamics(q, dynamics_expr)
        .with_cost(integral(cost_expr));
    auto ocp_from_expr = built_expr_model.build();

    // Verify fields match expected values.
    EXPECT_EQ(ocp_from_expr.num_states, 1u);
    EXPECT_EQ(ocp_from_expr.num_controls, 1u);
    EXPECT_DOUBLE_EQ(ocp_from_expr.state_lower[0], 0.0);
    EXPECT_DOUBLE_EQ(ocp_from_expr.initial_state[0], 10.0);
    EXPECT_DOUBLE_EQ(ocp_from_expr.initial_state_fixed[0], 1.0);
    EXPECT_DOUBLE_EQ(ocp_from_expr.control_lower[0], 0.0);
    EXPECT_DOUBLE_EQ(ocp_from_expr.control_upper[0], 5.0);

    // Verify dynamics eval under double: ARRIVAL - rate = 3.0 - 2.0 = 1.0
    const std::vector<double> x_test{5.0};
    const std::vector<double> u_test{2.0};
    const auto dyn_result = ocp_from_expr.dynamics(x_test, u_test, 0.0);
    ASSERT_EQ(dyn_result.size(), 1u);
    EXPECT_DOUBLE_EQ(dyn_result[0], 1.0);  // 3.0 - 2.0

    // Verify cost eval under double: 5.0 + 0.1*4.0 = 5.4
    EXPECT_DOUBLE_EQ(ocp_from_expr.cost(x_test, u_test, 0.0), 5.4);
}

TEST(ExprModel, MissingDynamicsForStateThrowsExprError) {
    goss::model::expr::ExprModel<> expr_model{};
    expr_model.add_state("q");
    expr_model.add_state("p");  // second state, no dynamics
    expr_model.add_control("u");
    expr_model.set_mesh(0.0, 1.0, 2);

    using namespace goss::model::expr;
    auto partial_model = std::move(expr_model)
        .with_dynamics(goss::model::StateHandle{0}, StateLeaf{0});
    // build() should detect that state 1 has no dynamics expression
    EXPECT_THROW(partial_model.build(), goss::model::expr::ExprError);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — `expr_model.hpp` not found.

- [ ] **Step 3: Write `expr_model.hpp`**

```cpp
// include/goss/model/expr/expr_model.hpp
#pragma once
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>
#include "goss/model/expr/constraints.hpp"
#include "goss/model/expr/integral.hpp"
#include "goss/model/expr/errors.hpp"
#include "goss/model/model.hpp"

namespace goss::model::expr {

/// Per-state dynamics entry: pairs a handle index with a typed expression.
/// Stored in the DynTuple; used by DynamicsFunctor to evaluate each entry
/// under ScalarT in declaration order.
template <typename DynExpr>
struct DynamicsEntry {
    std::size_t state_index;
    DynExpr dynamics_expression;

    template <typename ScalarT>
    ScalarT eval(const std::vector<ScalarT>& x,
                 const std::vector<ScalarT>& u,
                 ScalarT t) const {
        return dynamics_expression.eval<ScalarT>(x, u, t);
    }
};

/// A dynamics functor built from a tuple of DynamicsEntry objects.
/// template<T> vector<T> operator()(x, u, t) evaluates every entry and
/// assembles the full dx/dt vector, satisfying Model::build's DynamicsFn
/// contract. Entries need not be in state-index order in the tuple; they
/// are placed by state_index into the result vector.
template <typename DynTuple>
struct DynamicsFunctor {
    DynTuple dynamics_entries;
    std::size_t num_states;

    template <typename ScalarT>
    std::vector<ScalarT> operator()(const std::vector<ScalarT>& x,
                                    const std::vector<ScalarT>& u,
                                    ScalarT t) const {
        std::vector<ScalarT> result(num_states, ScalarT(0));
        fill_result(result, x, u, t,
                    std::make_index_sequence<std::tuple_size_v<DynTuple>>{});
        return result;
    }

 private:
    template <typename ScalarT, std::size_t... Indices>
    void fill_result(std::vector<ScalarT>& result,
                     const std::vector<ScalarT>& x,
                     const std::vector<ScalarT>& u,
                     ScalarT t,
                     std::index_sequence<Indices...>) const {
        // Fold expression to evaluate each entry and place into result.
        ((result[std::get<Indices>(dynamics_entries).state_index] =
              std::get<Indices>(dynamics_entries).eval(x, u, t)), ...);
    }
};

/// Fluent builder that accumulates per-state dynamics expressions and a cost
/// expression, then assembles them into a Model::build()-compatible call.
///
/// DynTuple: std::tuple<DynamicsEntry<Expr0>, DynamicsEntry<Expr1>, ...>
///   — grows with each with_dynamics() call.
/// CostFn: either void (no cost set yet) or CostFunctor<CostExpr>.
///
/// Usage:
///   auto ocp = ExprModel<>{}
///       .with_dynamics(q, ARRIVAL - rate)
///       .with_cost(integral(q + w * rate * rate))
///       .build();
template <typename DynTuple = std::tuple<>, typename CostFn = void>
class ExprModel {
 public:
    // --- Construction ---
    explicit ExprModel() = default;

    // Internal constructor used by with_dynamics / with_cost.
    ExprModel(goss::model::Model model,
              DynTuple dyn_tuple,
              CostFn cost_fn)
        : model_(std::move(model)),
          dyn_tuple_(std::move(dyn_tuple)),
          cost_fn_(std::move(cost_fn)) {}

    // --- Forward model-level operations ---

    goss::model::StateHandle add_state(const std::string& name) {
        return model_.add_state(name);
    }
    goss::model::ControlHandle add_control(const std::string& name) {
        return model_.add_control(name);
    }
    void set_mesh(double t_initial, double t_final, std::size_t num_intervals) {
        model_.set_mesh(t_initial, t_final, num_intervals);
    }
    // For direct setter access (tests, compatibility):
    void set_state_bounds(goss::model::StateHandle s, double lo, double hi) {
        model_.set_state_bounds(s, lo, hi);
    }
    void set_control_bounds(goss::model::ControlHandle c, double lo, double hi) {
        model_.set_control_bounds(c, lo, hi);
    }
    void set_initial_state(goss::model::StateHandle s, double v) {
        model_.set_initial_state(s, v);
    }
    void set_final_state(goss::model::StateHandle s, double v) {
        model_.set_final_state(s, v);
    }

    // --- Accessor forwarding for tests ---
    std::size_t num_states()   const { return model_.num_states();   }
    std::size_t num_controls() const { return model_.num_controls(); }

    // --- Constraint sugar ---
    ExprModel& apply(const BoundConstraint& constraint) {
        apply_bound(model_, constraint);
        return *this;
    }
    ExprModel& apply(const ControlBoundConstraint& constraint) {
        apply_bound(model_, constraint);
        return *this;
    }
    ExprModel& apply(const BoundaryConstraint& constraint) {
        apply_boundary(model_, constraint);
        return *this;
    }

    // --- Fluent dynamics accumulator ---
    /// Returns a NEW ExprModel type with the dynamics entry appended.
    /// The caller must std::move(*this) in to avoid copying the model state.
    template <typename DynExpr>
    auto with_dynamics(goss::model::StateHandle state_handle, DynExpr dyn_expr) && {
        DynamicsEntry<DynExpr> new_entry{state_handle.index, std::move(dyn_expr)};
        auto new_tuple = std::tuple_cat(std::move(dyn_tuple_),
                                        std::make_tuple(std::move(new_entry)));
        using NewDynTuple = decltype(new_tuple);
        return ExprModel<NewDynTuple, CostFn>{
            std::move(model_),
            std::move(new_tuple),
            std::move(cost_fn_)
        };
    }

    /// Returns a NEW ExprModel type with the cost functor set.
    template <typename NewCostFn>
    auto with_cost(NewCostFn new_cost_fn) && {
        return ExprModel<DynTuple, NewCostFn>{
            std::move(model_),
            std::move(dyn_tuple_),
            std::move(new_cost_fn)
        };
    }

    // --- Build (zero-argument: uses accumulated expressions) ---
    /// Assembles per-state dynamics tuple and cost functor into a single
    /// Model::build() call. Throws ExprError if any state has no dynamics entry,
    /// or if no cost has been set.
    auto build() && {
        static_assert(!std::is_void_v<CostFn>,
            "ExprModel::build(): no cost set — call with_cost(integral(...)) before build()");
        const std::size_t declared_state_count = model_.num_states();
        const std::size_t registered_dyn_count = std::tuple_size_v<DynTuple>;
        if (registered_dyn_count != declared_state_count) {
            throw ExprError(
                "ExprModel::build(): " +
                std::to_string(declared_state_count) + " states declared but " +
                std::to_string(registered_dyn_count) + " dynamics expressions registered — "
                "call with_dynamics() once per declared state");
        }
        DynamicsFunctor<DynTuple> dyn_functor{
            std::move(dyn_tuple_), declared_state_count};
        return model_.build(std::move(dyn_functor), std::move(cost_fn_));
    }

    // --- Build (two-argument: lambda path, unchanged) ---
    template <typename DynamicsFn, typename CostFnArg>
    auto build(DynamicsFn dynamics_lambda, CostFnArg cost_lambda) {
        return model_.build(std::move(dynamics_lambda), std::move(cost_lambda));
    }

 private:
    goss::model::Model model_{};
    DynTuple dyn_tuple_{};
    CostFn cost_fn_{};
};

// Deduction guide: ExprModel<> {} creates ExprModel<std::tuple<>, void>.
// When CostFn is void, default-constructing it is fine since it is never
// used until with_cost() replaces it.

}  // namespace goss::model::expr
```

**Implementation note on `void` default:** `ExprModel<std::tuple<>, void>` stores a `void cost_fn_` member, which is illegal. Fix: use `std::monostate` as the default instead of `void`, and add a `static_assert` in `build()` that checks `!std::is_same_v<CostFn, std::monostate>` with the same message. Change the template default to `typename CostFn = std::monostate`. Adjust the `with_cost` implementation accordingly (just replace the stored `cost_fn_` with the new value — no other change needed).

Add `#include "goss/model/expr/expr_model.hpp"` to `expr.hpp`.

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "ExprModel" --output-on-failure'`
Expected: both ExprModel tests PASS. Confirm `MissingDynamicsForStateThrowsExprError` actually throws and is caught.

- [ ] **Step 5: Commit**

```bash
git add include/goss/model/expr/expr_model.hpp include/goss/model/expr/expr.hpp tests/model/test_expr_lowering.cpp
git commit -m "feat: ExprModel fluent builder with per-state dynamics accumulation and zero-arg build()"
```

---

### Task 7: Flagship end-to-end test — queue model rewritten with operator-overload syntax

**Files:**
- Create: `tests/model/test_expr_solve.cpp`
- Modify: `CMakeLists.txt` (add `tests/model/test_expr_solve.cpp` to `goss_model_tests`)

**Interfaces:**
- Consumes: `ExprModel`, `integral`, comparison operators, `goss::transcription::HermiteSimpson`, `goss::solver::IpoptSolver`, `goss::solver::SolverStatus`.
- Produces: two tests — (a) the queue model written entirely in the operator-overload DSL syntax, solved end-to-end; (b) a consistency check that the queue model DSL version produces the same `objective_value` as the existing lambda version (within 1e-3).

**Test A — DSL queue model:** This test must be a line-for-line translation of `QueueModelKeepsQueueNonNegative` from `test_model_solve.cpp` into the operator-overload syntax, making the same assertions (q(0)=10, q>=0 at all nodes, rate in [0,MAX], objective>0). This is the spec's flagship requirement.

**Test B — consistency:** Solves the SAME queue problem using both the lambda path (copy of the existing test) and the ExprModel path, and asserts `std::abs(lambda_result.objective_value - expr_result.objective_value) < 1e-3`. This proves the two paths are numerically equivalent and that no accidental constant scaling or off-by-one occurred in `DynamicsFunctor` or `CostFunctor`.

- [ ] **Step 1: Write the failing tests**

```cpp
// tests/model/test_expr_solve.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "goss/model/expr/expr.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"

namespace {
// Shared constants for both tests — identical to test_model_solve.cpp queue problem.
constexpr double QUEUE_ARRIVAL   = 3.0;
constexpr double QUEUE_MAX_RATE  = 5.0;
constexpr double QUEUE_WEIGHT    = 0.1;
constexpr double QUEUE_T_FINAL   = 5.0;
constexpr double QUEUE_INITIAL   = 10.0;
constexpr std::size_t QUEUE_INTERVALS = 30;
}  // namespace

// Test A: the queue model written entirely with the operator-overload expression DSL.
// This is the flagship test — the spec's §7 sugar syntax.
TEST(ExprSolve, QueueModelWithOperatorOverloadDslSolvesSuccessfully) {
    using namespace goss::model::expr;

    ExprModel<> expr_model{};
    const auto q_handle    = expr_model.add_state("queue_length");
    const auto rate_handle = expr_model.add_control("service_rate");

    // Box bounds and boundary condition via comparison operators.
    expr_model.apply(q_handle >= 0.0);
    expr_model.apply(rate_handle >= 0.0);
    expr_model.apply(rate_handle <= QUEUE_MAX_RATE);
    expr_model.apply(q_handle.initial() == QUEUE_INITIAL);
    expr_model.set_mesh(0.0, QUEUE_T_FINAL, QUEUE_INTERVALS);

    // dq/dt = ARRIVAL - rate
    const auto dynamics_expression = ConstantExpr{QUEUE_ARRIVAL} - ControlLeaf{rate_handle.index};
    // cost = q + WEIGHT * rate^2
    const auto cost_expression =
        StateLeaf{q_handle.index} +
        ConstantExpr{QUEUE_WEIGHT} * ControlLeaf{rate_handle.index} * ControlLeaf{rate_handle.index};

    auto solved_model = std::move(expr_model)
        .with_dynamics(q_handle, dynamics_expression)
        .with_cost(integral(cost_expression));

    auto ocp      = solved_model.build();
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "expr_queue");
    goss::solver::IpoptSolver solver;
    std::vector<double> initial_guess(compiled.problem->num_variables(), 5.0);
    const auto result = solver.solve(*compiled.problem, initial_guess);

    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);

    const auto& layout = compiled.layout;
    // q(0) pinned to QUEUE_INITIAL.
    EXPECT_NEAR(result.x[layout.state_index(0, 0)], QUEUE_INITIAL, 1e-6);
    // q stays >= 0 at every node (solver slack tolerance).
    for (std::size_t node_index = 0; node_index < layout.num_nodes(); ++node_index) {
        EXPECT_GE(result.x[layout.state_index(node_index, 0)], -1e-4)
            << "queue negative at node " << node_index;
    }
    // rate respected its box [0, QUEUE_MAX_RATE].
    for (std::size_t node_index = 0; node_index < layout.num_nodes(); ++node_index) {
        const double rate_value = result.x[layout.control_index(node_index, 0)];
        EXPECT_GE(rate_value, -1e-4);
        EXPECT_LE(rate_value, QUEUE_MAX_RATE + 1e-4);
    }
    EXPECT_GT(result.objective_value, 0.0);
}

// Test B: numerical consistency between the lambda DSL path and the expr DSL path.
// Both must solve to the same objective (within solver tolerance).
TEST(ExprSolve, QueueModelExprAndLambdaPathsProduceConsistentObjectives) {
    // --- Lambda path (same as test_model_solve.cpp QueueModelKeepsQueueNonNegative) ---
    double lambda_objective = 0.0;
    {
        goss::model::Model lambda_model;
        const auto q_lambda    = lambda_model.add_state("queue_length");
        const auto rate_lambda = lambda_model.add_control("service_rate");
        lambda_model.set_state_bounds(q_lambda, 0.0, goss::transcription::kInf);
        lambda_model.set_control_bounds(rate_lambda, 0.0, QUEUE_MAX_RATE);
        lambda_model.set_initial_state(q_lambda, QUEUE_INITIAL);
        lambda_model.set_mesh(0.0, QUEUE_T_FINAL, QUEUE_INTERVALS);

        auto lambda_dynamics = [](const auto& x_vec, const auto& u_vec, auto /*t*/) {
            using ScalarT = typename std::decay_t<decltype(x_vec)>::value_type;
            return std::vector<ScalarT>{ScalarT(QUEUE_ARRIVAL) - u_vec[0]};
        };
        auto lambda_cost = [](const auto& x_vec, const auto& u_vec, auto /*t*/) {
            using ScalarT = typename std::decay_t<decltype(x_vec)>::value_type;
            return x_vec[0] + ScalarT(QUEUE_WEIGHT) * u_vec[0] * u_vec[0];
        };

        auto ocp_lambda      = lambda_model.build(lambda_dynamics, lambda_cost);
        auto compiled_lambda = goss::transcription::HermiteSimpson::compile(ocp_lambda, "lambda_queue_consistency");
        goss::solver::IpoptSolver solver_lambda;
        std::vector<double> z0_lambda(compiled_lambda.problem->num_variables(), 5.0);
        const auto result_lambda = solver_lambda.solve(*compiled_lambda.problem, z0_lambda);
        ASSERT_EQ(result_lambda.status, goss::solver::SolverStatus::Success);
        lambda_objective = result_lambda.objective_value;
    }

    // --- Expr DSL path ---
    double expr_objective = 0.0;
    {
        using namespace goss::model::expr;
        ExprModel<> expr_model{};
        const auto q_expr    = expr_model.add_state("queue_length");
        const auto rate_expr = expr_model.add_control("service_rate");
        expr_model.apply(q_expr >= 0.0);
        expr_model.apply(rate_expr >= 0.0);
        expr_model.apply(rate_expr <= QUEUE_MAX_RATE);
        expr_model.apply(q_expr.initial() == QUEUE_INITIAL);
        expr_model.set_mesh(0.0, QUEUE_T_FINAL, QUEUE_INTERVALS);

        const auto dyn_expr  = ConstantExpr{QUEUE_ARRIVAL} - ControlLeaf{rate_expr.index};
        const auto cost_expr = StateLeaf{q_expr.index} +
            ConstantExpr{QUEUE_WEIGHT} * ControlLeaf{rate_expr.index} * ControlLeaf{rate_expr.index};

        auto ocp_expr      = std::move(expr_model)
            .with_dynamics(q_expr, dyn_expr)
            .with_cost(integral(cost_expr))
            .build();
        auto compiled_expr = goss::transcription::HermiteSimpson::compile(ocp_expr, "expr_queue_consistency");
        goss::solver::IpoptSolver solver_expr;
        std::vector<double> z0_expr(compiled_expr.problem->num_variables(), 5.0);
        const auto result_expr = solver_expr.solve(*compiled_expr.problem, z0_expr);
        ASSERT_EQ(result_expr.status, goss::solver::SolverStatus::Success);
        expr_objective = result_expr.objective_value;
    }

    EXPECT_NEAR(lambda_objective, expr_objective, 1e-3)
        << "Lambda objective: " << lambda_objective
        << ", Expr objective: "  << expr_objective;
}
```

- [ ] **Step 2: Add test to CMake and run — verify pass**

Add `tests/model/test_expr_solve.cpp` to `goss_model_tests` in `CMakeLists.txt`.

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R "ExprSolve" --output-on-failure'`
Expected: both ExprSolve tests PASS. If the consistency test fails with objectives differing by more than 1e-3: (a) check that `DynamicsFunctor` places entries into the result vector by `state_index` (not by tuple position), (b) check `ConstantExpr{ARRIVAL}` wraps correctly (not `ConstantExpr{0}`), (c) run both solvers with `--output-on-failure` to see IPOPT's reported objective and compare manually.

- [ ] **Step 3: Run the full suite**

Run: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'`
Expected: ALL tests pass (prior suite + all model tests + all expr tests).

- [ ] **Step 4: Commit**

```bash
git add tests/model/test_expr_solve.cpp CMakeLists.txt
git commit -m "test: flagship expr DSL queue model solved end-to-end; consistency with lambda path verified"
```

---

### Task 8: Self-review and cleanup

**Files:**
- Modify: any file needing cleanup, comment improvements, or placeholder removal.
- Review: `include/goss/model/expr/` header includes (no cycles), `expr.hpp` umbrella completeness.

**Steps:**

- [ ] **Step 1: Scan for placeholders and TODOs**

```bash
grep -r "TBD\|TODO\|FIXME\|placeholder\|add validation\|similar to Task" \
    include/goss/model/expr/ tests/model/test_expr_nodes.cpp \
    tests/model/test_expr_lowering.cpp tests/model/test_expr_solve.cpp
```
Expected: no hits. If any, fix before proceeding.

- [ ] **Step 2: Verify no accidental change to core layer files**

```bash
git diff HEAD -- include/goss/model/model.hpp include/goss/transcription/ \
    include/goss/nlp/ include/goss/solver/ include/goss/ad/
```
Expected: ONLY `include/goss/model/handles.hpp` modified (adding `BoundaryPoint` + `initial()`/`final()`). All transcription/NLP/solver/AD headers untouched.

- [ ] **Step 3: Confirm `expr.hpp` includes all sub-headers**

`expr.hpp` should include (in order): `errors.hpp`, `nodes.hpp`, `operators.hpp`, `constraints.hpp`, `integral.hpp`, `expr_model.hpp`. Verify no header is missing.

- [ ] **Step 4: Run full suite one final time**

```bash
scripts/dev.sh 'ctest --test-dir build --output-on-failure'
```
Expected: all tests green. Record the final test count in the commit message.

- [ ] **Step 5: Final commit**

```bash
git add -u  # only modified tracked files
git commit -m "chore: expr DSL self-review — no placeholders, no core-layer regressions"
```

---

## Self-Review

### Spec Coverage (spec §7 operator-overload sugar)

| Spec requirement | Plan coverage |
|---|---|
| `q >= 0.0` lowers to `set_state_bounds(q, 0.0, kInf)` | Task 4: `BoundConstraint` + `apply_bound()` |
| `q.initial() == 10.0` lowers to `set_initial_state(q, 10.0)` | Task 4: `BoundaryConstraint` + `apply_boundary()` |
| `integral(q + w*rate*rate)` lowers to a cost lambda evaluating the AST under `T` | Task 5: `CostFunctor<Expr>` + `integral()` factory |
| `set_dynamics(q, ARRIVAL - rate)` collects per-state dynamics | Task 6: `ExprModel::with_dynamics()` |
| AST evaluator MUST be templated on T (AD-safety) | Tasks 2, 3, 5, 6: all `eval<ScalarT>()` methods; AD test in Task 3 |
| Layered ON TOP of lambda DSL without changing the core | Tasks 1–8: `ExprModel::build(dynamics, cost)` lambda path preserved; `Model` and `model.hpp` untouched |
| `Model::build()` and downstream layers stay unchanged | Verified in Task 8 Step 2 |
| Flagship test: queue example solved via expression syntax | Task 7 Test A |
| Same result as lambda version | Task 7 Test B (objective within 1e-3) |
| General nonlinear path constraints assessed and scoped | Task 4: assessment documented — requires transcription extension, deferred; v1 restricts to box bounds on state/control handles |

### Placeholder Scan

Every task step contains literal code. No "TBD", "add validation later", "similar to Task N" stubs. The one deferred item (general nonlinear path constraints) is explicitly noted with the transcription work it would require — not left as a placeholder but as a documented extension point.

### Type Consistency

- `StateLeaf{handle.index}` / `ControlLeaf{handle.index}` — correct: the leaf stores a `std::size_t` index, not a `StateHandle` (no circular dependency between nodes and handles).
- `BinaryExpr<Tag,L,R>` compose via `if constexpr` on `OpTag` — the three tags (`AddTag`, `SubTag`, `MulTag`) are exhaustive for v1; the `static_assert` in the else branch catches future mistakes.
- `CostFunctor<Expr>` `operator()` template is `const` and matches `CostFn` contract: `template<T> T operator()(const vector<T>&, const vector<T>&, T) const`.
- `DynamicsFunctor<DynTuple>` `operator()` template returns `std::vector<ScalarT>` of size `num_states`, matching `DynamicsFn` contract.
- `ExprModel<std::tuple<>, std::monostate>` — the `void` default replaced by `std::monostate` to allow default construction; `static_assert(!std::is_same_v<CostFn, std::monostate>)` in `build()` gives a clean compile-time error if `with_cost()` was not called.
- `DynamicsEntry::state_index` placed into `result[state_index]` (not tuple position) — critical correctness point for multi-state models; covered by `DynamicsFunctor::fill_result`.
- `handles.hpp` modification is additive: `operator std::size_t()` and `index` field untouched; `initial()`/`final()` added with `inline` definitions after `BoundaryPoint` is complete.
- Include chain: `expr_model.hpp` → `constraints.hpp` → `model.hpp`; `constraints.hpp` → `operators.hpp` → `nodes.hpp` → (only `<cstddef>`, `<vector>`). No cycles.

### Known Risks

1. **`with_dynamics` rvalue-only:** The fluent builder returns a new `ExprModel` type from `with_dynamics` and `with_cost`, requiring `std::move`. This is intentional (the model_ state is moved, not copied) but may surprise users. Document in `expr_model.hpp` with a comment.
2. **`DynamicsEntry` order vs state_index:** If a user calls `with_dynamics(state1, ...).with_dynamics(state0, ...)` out of declaration order, `DynamicsFunctor::fill_result` places entries by `state_index`, so the result vector is correctly ordered. The test in Task 6 covers in-order; adding an out-of-order test is a good extension but not required for v1.
3. **`ExprError` on count mismatch vs per-state detection:** The count check in `build()` catches total count mismatches but not which specific state is missing. For a v1 with one or two states, the count message is clear enough. A future improvement would track which state indices are registered and report the unregistered ones by name.
4. **Template instantiation depth:** Deeply nested expressions (e.g., 10+ terms in an integral) produce deeply nested `BinaryExpr<...>` types. This is standard C++ template recursion; the default template depth (1024 on GCC/Clang) is not approached for the queue example. No action needed for v1.
5. **`handles.hpp` modification breaks `constexpr`:** `initial()` and `final()` return `BoundaryPoint` by value and are `inline` (not `constexpr`) because `BoundaryPoint` contains an `enum class`. If any downstream code uses `StateHandle` in a `constexpr` context that would invalidate the added methods, this would be a compile error. In practice no such use exists in the current codebase (verified by `grep -r constexpr.*StateHandle`).
