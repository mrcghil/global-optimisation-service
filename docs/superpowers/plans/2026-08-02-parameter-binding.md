# Parameter Binding (Compile-Once) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a model declare named *parameters* (values that vary per run — arrival rate, a cost weight, an initial state) so the AD tape is recorded and JIT-compiled ONCE, and each subsequent solve only injects a fresh parameter vector — no recompile.

**Architecture:** Add a parameter-binding capability at the AD backend (`ad/`), expose it through `NLPProblem` (`nlp/`), thread an optional parameter vector into the transcription packed functors (`transcription/`), let `Model` declare parameters and produce a validation artifact (`model/`), and add a `sim/` helper that validates-then-binds a parameter set with explicit error messages. Upper layers depend only on a stable `set_parameters`/validation abstraction; the concrete binding mechanism (CppAD dynamic parameters, with a pinned-variable fallback) is hidden inside `ad/`.

**Tech Stack:** C++17, CppAD + CppADCodeGen (JIT), IPOPT/NLopt, GoogleTest, CMake.

## Global Constraints

- **C++ standard:** C++17 (`CMAKE_CXX_STANDARD 17`, `CMakeLists.txt:5`). No C++20 features.
- **Header hygiene:** no third-party solver/AD library type may appear in a public header outside its own `.cpp`/impl (e.g. IPOPT confined to `src/solver/*.cpp`; CppADCodeGen types only under `goss::ad::detail`). Follow existing precedent in `ad_backend.hpp` / `ipopt_solver.hpp`.
- **Error handling (org standard):** specific exception types, meaningful messages naming the offending symbol, no silent catch-alls. Reuse the per-layer error types: `ad::ADError`, `nlp::NLPError`, `model::ModelError`, `transcription::TranscriptionError`, `sim::SimError`.
- **Naming (user preference):** verbose descriptive names; type annotations everywhere the language allows.
- **Additive, not rewrite (spec §1 core goal):** existing 3-argument dynamics/cost functors `(x, u, t)` MUST keep compiling and passing unchanged. Parameter support is opt-in via a 4-argument overload `(x, u, p, t)`.
- **JIT model names must be unique** across concurrently-built models/processes — the `model_name` drives the generated `.so` filename (`cppadcg_backend.hpp:64-70`).
- **Build/test:** configure/build under `build/`; run a target with `ctest` or the target binary directly. Every task ends green.

---

### Task 1: Spike — pin down the compile-once parameter mechanism in `ad/`

This task de-risks the entire plan. It decides HOW a parameter is injected into an already-JIT-compiled model without re-taping. It produces a passing standalone test and a one-paragraph decision note; it does NOT yet change the public backend.

**Two candidate mechanisms (spike must pick one and prove it):**
- **(Primary) CppAD dynamic parameters:** record with `CppAD::Independent(ax, adynamic)`; the compiled CppADCodeGen model evaluates `f`, Jacobian, Hessian as functions of `x` with the dynamic vector injected per call.
- **(Fallback, zero exotic-API risk) Pinned decision variables:** append the parameters to the independent vector `z`, and "set a parameter" = set that variable's lower==upper bound to the value. Sparsity is stable; the solver cannot move a fixed-bound variable. Guaranteed to work with the current backend with no CppADCodeGen dynamic-parameter support.

Either way, the *upper-layer abstraction is identical* (`num_parameters()` + `set_parameters(p)`), so downstream tasks are unaffected by which wins.

**Files:**
- Create: `tests/ad/test_dynamic_param_spike.cpp`
- Modify: `CMakeLists.txt:51-64` (add the spike source to `goss_ad_tests`)
- Create: `docs/superpowers/plans/notes/2026-08-02-param-mechanism-decision.md` (the decision record)

**Interfaces:**
- Consumes: `CppAD` / `CppAD::cg` (via `<cppad/cg.hpp>`), the existing `detail::compile_and_load` pattern in `src/ad/cppadcg_backend.cpp:12`.
- Produces: a proven code path (recording + injection) that Task 2 will wrap. Records the chosen mechanism name in the decision note.

- [ ] **Step 1: Write the failing spike test (behavioral contract)**

```cpp
// tests/ad/test_dynamic_param_spike.cpp
#include <gtest/gtest.h>
#include <cppad/cg.hpp>
#include <vector>

// Goal: record f(x; p) = p0 * x0 * x0 ONCE, JIT-compile ONCE, then evaluate at
// two different parameter values WITHOUT re-recording/re-compiling.
// f  = p0 * x0^2         -> value depends on p0
// df/dx0 = 2*p0*x0       -> jacobian depends on p0
// This proves compile-once parameter injection end-to-end.
namespace {
using CGD  = CppAD::cg::CG<double>;
using ADCG = CppAD::AD<CGD>;
}  // namespace

TEST(DynamicParamSpike, ValueAndJacobianTrackInjectedParameterWithoutRecompile) {
    const std::size_t num_vars = 1;
    const std::size_t num_params = 1;

    std::vector<ADCG> ax(num_vars);
    std::vector<ADCG> ap(num_params);
    ax[0] = 1.0;
    ap[0] = 1.0;
    // Primary mechanism under test: dynamic-parameter recording.
    CppAD::Independent(ax, ap);
    std::vector<ADCG> ay(1);
    ay[0] = ap[0] * ax[0] * ax[0];
    CppAD::ADFun<CGD> fun(ax, ay);
    fun.optimize();

    // Generate + JIT-compile ONCE via the same pipeline as compile_and_load.
    CppAD::cg::ModelCSourceGen<double> source_gen(fun, "spike_model");
    source_gen.setCreateForwardZero(true);
    source_gen.setCreateSparseJacobian(true);
    CppAD::cg::ModelLibraryCSourceGen<double> library_gen(source_gen);
    CppAD::cg::GccCompiler<double> compiler;
    CppAD::cg::DynamicModelLibraryProcessor<double> processor(library_gen, "spike_model");
    auto library = processor.createDynamicLibrary(compiler);
    auto model = library->model("spike_model");
    ASSERT_TRUE(model);

    // The compiled model reports one dynamic parameter (proves codegen kept it).
    ASSERT_EQ(model->Domain(), num_vars);

    // Inject p0 = 3.0, evaluate at x0 = 2.0  ->  f = 3*4 = 12 ; df/dx0 = 2*3*2 = 12
    model->new_dynamic(std::vector<double>{3.0});
    std::vector<double> y1 = model->ForwardZero(std::vector<double>{2.0});
    EXPECT_NEAR(y1[0], 12.0, 1e-12);

    // Inject p0 = 5.0 on the SAME compiled model -> f = 5*4 = 20
    model->new_dynamic(std::vector<double>{5.0});
    std::vector<double> y2 = model->ForwardZero(std::vector<double>{2.0});
    EXPECT_NEAR(y2[0], 20.0, 1e-12);
}
```

- [ ] **Step 2: Add the spike TU to the AD test target and configure**

In `CMakeLists.txt`, add `tests/ad/test_dynamic_param_spike.cpp` to the `goss_ad_tests` executable source list (`CMakeLists.txt:51-57`).

Run: `cmake -S . -B build && cmake --build build --target goss_ad_tests`
Expected: compiles; test present.

- [ ] **Step 3: Run the spike**

Run: `cd build && ctest -R DynamicParamSpike --output-on-failure`

Two outcomes:
- **PASS** → primary mechanism (CppAD dynamic parameters via `new_dynamic` + `ForwardZero`/`SparseJacobian`) is viable. Proceed with it in Task 2.
- **FAIL** (compile error on `new_dynamic`/`Domain`, or wrong values) → the pinned CppAD/CppADCodeGen version does not support dynamic parameters through GenericModel. Switch to the fallback: rewrite the spike to append the parameter as an extra independent variable and prove that fixing its value at eval time (passed in the `x` vector) yields the same value/Jacobian behavior. Both are compile-once.

- [ ] **Step 4: Record the decision**

Write `docs/superpowers/plans/notes/2026-08-02-param-mechanism-decision.md` stating: chosen mechanism, the exact GenericModel calls used (e.g. `model->new_dynamic(p)` before each `ForwardZero`/`SparseJacobian`/`SparseHessian`), and — if fallback — how parameters map to appended pinned variables. Task 2 implements exactly this.

- [ ] **Step 5: Commit**

```bash
git add tests/ad/test_dynamic_param_spike.cpp CMakeLists.txt docs/superpowers/plans/notes/2026-08-02-param-mechanism-decision.md
git commit -m "spike(ad): prove compile-once parameter injection mechanism"
```

---

### Task 2: `ad/` — parameterized backend capability

Extend the AD backend so it can record a functor of `(z, p)` and re-inject `p` on the already-compiled model. Existing single-argument recording is untouched.

**Files:**
- Modify: `include/goss/ad/ad_backend.hpp:16-40` (add `num_parameters`, `set_parameters` to the interface with a default no-op so existing subclasses/mocks still compile)
- Modify: `include/goss/ad/cppadcg_backend.hpp` (new constructor overload recording `(z,p)`; store param count; implement `set_parameters`)
- Modify: `src/ad/cppadcg_backend.cpp:12-67` (teach `compile_and_load` to enable dynamic parameters per the Task 1 decision; capture sparsity with a representative parameter set)
- Test: `tests/ad/test_parametric_backend.cpp`
- Modify: `CMakeLists.txt:51-57`

**Interfaces:**
- Consumes: the mechanism proven in Task 1.
- Produces:
  - `std::size_t ad::ADBackend::num_parameters() const` — default returns `0`.
  - `void ad::ADBackend::set_parameters(const std::vector<double>& parameter_values)` — default throws `ADError("backend has no parameters")` if `parameter_values` is non-empty, else no-op.
  - New `CppADCGBackend` constructor:
    ```cpp
    template <typename F>
    CppADCGBackend(const F& function, std::size_t input_size,
                   std::size_t parameter_size,
                   const std::vector<double>& parameter_defaults,
                   const std::string& model_name = "goss_model");
    ```
    where `F` has signature `std::vector<T> operator()(const std::vector<T>& z, const std::vector<T>& p)`.
  - `CppADCGBackend::num_parameters()` returns `parameter_size`; `set_parameters(p)` injects `p` (throws `ADError` on size mismatch with an explicit message).

- [ ] **Step 1: Add interface methods with safe defaults (failing contract test)**

Add to `ad_backend.hpp` (non-pure, so existing subclasses compile unchanged):

```cpp
    /// Number of injectable parameters (values fixed per-evaluation, held
    /// constant across the solver's x-iterations). Zero unless the backend was
    /// recorded with a parameter functor.
    virtual std::size_t num_parameters() const { return 0; }

    /// Injects the parameter vector for all subsequent eval/jacobian/hessian
    /// calls. Throws ADError if parameter_values.size() != num_parameters().
    /// Default: accepts only an empty vector (no parameters).
    virtual void set_parameters(const std::vector<double>& parameter_values) {
        if (!parameter_values.empty())
            throw ADError("set_parameters: backend has no parameters but " +
                          std::to_string(parameter_values.size()) + " were provided");
    }
```

Test in `tests/ad/test_parametric_backend.cpp`:

```cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/ad/cppadcg_backend.hpp"
#include "goss/ad/errors.hpp"

namespace {
// f(z; p) = p0 * z0^2 + z1 ; grad_z = [2*p0*z0, 1]
struct ParamQuadratic {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& z, const std::vector<T>& p) const {
        return { p[0] * z[0] * z[0] + z[1] };
    }
};
}  // namespace

TEST(ParametricBackend, EvalTracksInjectedParameter) {
    goss::ad::CppADCGBackend backend(
        ParamQuadratic{}, /*input_size=*/2, /*parameter_size=*/1,
        /*parameter_defaults=*/{1.0}, "param_quad_eval");
    ASSERT_EQ(backend.num_parameters(), 1u);

    backend.set_parameters({3.0});
    EXPECT_NEAR(backend.eval({2.0, 7.0})[0], 3.0 * 4.0 + 7.0, 1e-12);  // 19

    backend.set_parameters({5.0});
    EXPECT_NEAR(backend.eval({2.0, 7.0})[0], 5.0 * 4.0 + 7.0, 1e-12);  // 27
}
```

- [ ] **Step 2: Add spike TU wiring; run to verify FAIL**

Add `tests/ad/test_parametric_backend.cpp` to `goss_ad_tests` in `CMakeLists.txt:51-57`.

Run: `cmake --build build --target goss_ad_tests`
Expected: FAIL — no `CppADCGBackend` constructor taking `parameter_size`.

- [ ] **Step 3: Implement the parameterized constructor**

In `cppadcg_backend.hpp`, add the new constructor. It records with the Task-1 mechanism. For the dynamic-parameter path:

```cpp
template <typename F>
CppADCGBackend(const F& function, std::size_t input_size,
               std::size_t parameter_size,
               const std::vector<double>& parameter_defaults,
               const std::string& model_name = "goss_model")
    : input_size_(input_size), parameter_size_(parameter_size) {
    if (parameter_defaults.size() != parameter_size)
        throw ADError("CppADCGBackend: parameter_defaults.size() (" +
                      std::to_string(parameter_defaults.size()) +
                      ") != parameter_size (" + std::to_string(parameter_size) + ")");

    std::vector<detail::ADCG> az(input_size);
    std::vector<detail::ADCG> ap(parameter_size);
    for (std::size_t i = 0; i < parameter_size; ++i) ap[i] = parameter_defaults[i];
    CppAD::Independent(az, ap);                 // az independent, ap dynamic
    std::vector<detail::ADCG> ay = function(az, ap);
    output_size_ = ay.size();
    CppAD::ADFun<detail::CGScalar> fun(az, ay);
    fun.optimize();
    compiled_ = detail::compile_and_load(fun, model_name);

    // Inject defaults so the first sparsity probe below sees a valid param set.
    compiled_.model->new_dynamic(parameter_defaults);

    // ... identical sparsity/permutation precompute as the existing constructor
    //     (copy the jacobian_sparsity_/hessian_sparsity_/jac_perm_/hess_perm_
    //      blocks verbatim from cppadcg_backend.hpp:88-226).
}
```

Add member `std::size_t parameter_size_ = 0;` and implement:

```cpp
std::size_t num_parameters() const override { return parameter_size_; }

void set_parameters(const std::vector<double>& parameter_values) override {
    if (parameter_values.size() != parameter_size_)
        throw ADError("set_parameters: expected " + std::to_string(parameter_size_) +
                      " parameters but got " + std::to_string(parameter_values.size()));
    compiled_.model->new_dynamic(parameter_values);   // per Task 1 decision
}
```

(If Task 1 chose the fallback: `parameter_size_` extra independent vars are appended; `set_parameters` stores the values into a member `param_values_` that `eval`/`eval_jacobian`/`eval_hessian` splice into the `x` passed to the compiled model. Implement whichever the note records — the tests above are mechanism-agnostic.)

Refactor the shared sparsity/permutation precompute into a private helper called by both constructors to avoid duplication (DRY).

- [ ] **Step 4: Run tests to verify PASS**

Run: `cd build && ctest -R ParametricBackend --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Add a Jacobian-tracks-parameter test and a bad-size test**

```cpp
TEST(ParametricBackend, JacobianTracksParameter) {
    goss::ad::CppADCGBackend backend(ParamQuadratic{}, 2, 1, {1.0}, "param_quad_jac");
    backend.set_parameters({4.0});
    // grad_z0 = 2*p0*z0 = 2*4*2 = 16 ; grad_z1 = 1
    auto jac = backend.eval_jacobian({2.0, 7.0});
    const auto& pattern = backend.jacobian_sparsity();
    ASSERT_EQ(pattern.size(), jac.size());
    for (std::size_t k = 0; k < pattern.size(); ++k) {
        if (pattern[k].second == 0) EXPECT_NEAR(jac[k], 16.0, 1e-9);
        if (pattern[k].second == 1) EXPECT_NEAR(jac[k], 1.0, 1e-9);
    }
}

TEST(ParametricBackend, SetParametersRejectsWrongSize) {
    goss::ad::CppADCGBackend backend(ParamQuadratic{}, 2, 1, {1.0}, "param_quad_badsz");
    EXPECT_THROW(backend.set_parameters({1.0, 2.0}), goss::ad::ADError);
}
```

Run: `cd build && ctest -R ParametricBackend --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/goss/ad/ad_backend.hpp include/goss/ad/cppadcg_backend.hpp src/ad/cppadcg_backend.cpp tests/ad/test_parametric_backend.cpp CMakeLists.txt
git commit -m "feat(ad): parameterized backend with compile-once parameter injection"
```

---

### Task 3: `nlp/` — parameter passthrough on NLPProblem

`NLPProblem` forwards parameter queries/injection to its backend. Eval methods are unchanged — parameters are injected out-of-band before a solve, and stay fixed across the solver's x-iterations.

**Files:**
- Modify: `include/goss/nlp/nlp_problem.hpp:29-59`
- Modify: `src/nlp/nlp_problem.cpp` (implement forwarding methods)
- Test: `tests/nlp/test_nlp_parameters.cpp`
- Modify: `CMakeLists.txt:71-74`

**Interfaces:**
- Consumes: `ad::ADBackend::num_parameters()`, `ad::ADBackend::set_parameters()` (Task 2).
- Produces:
  - `std::size_t nlp::NLPProblem::num_parameters() const` — forwards to backend.
  - `void nlp::NLPProblem::set_parameters(const std::vector<double>& parameter_values)` — forwards to backend (propagates `ADError`).

- [ ] **Step 1: Write the failing test**

```cpp
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
```

- [ ] **Step 2: Wire the test TU; run to verify FAIL**

Add `tests/nlp/test_nlp_parameters.cpp` to `goss_nlp_tests` (`CMakeLists.txt:71-74`).

Run: `cmake --build build --target goss_nlp_tests`
Expected: FAIL — `num_parameters`/`set_parameters` not members of `NLPProblem`.

- [ ] **Step 3: Add the forwarding declarations**

In `nlp_problem.hpp`, after `num_constraints()` (`nlp_problem.hpp:30`):

```cpp
    /// Number of injectable parameters (forwarded from the AD backend).
    std::size_t num_parameters() const;

    /// Injects the parameter vector for subsequent evaluations. Forwards to the
    /// backend; propagates ADError on size mismatch.
    void set_parameters(const std::vector<double>& parameter_values);
```

- [ ] **Step 4: Implement forwarding in the .cpp**

In `src/nlp/nlp_problem.cpp`:

```cpp
std::size_t NLPProblem::num_parameters() const {
    return backend_->num_parameters();
}

void NLPProblem::set_parameters(const std::vector<double>& parameter_values) {
    backend_->set_parameters(parameter_values);
}
```

- [ ] **Step 5: Run tests to verify PASS**

Run: `cd build && ctest -R NlpParameters --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/goss/nlp/nlp_problem.hpp src/nlp/nlp_problem.cpp tests/nlp/test_nlp_parameters.cpp CMakeLists.txt
git commit -m "feat(nlp): forward parameter count and injection to AD backend"
```

---

### Task 4: `model/` — parameter declaration + validation artifact

`Model` can declare parameters (name, default, bounds, optional predicate). `build()` records the parameter metadata into the `OcpProblem`, and a `ParameterValidator` artifact is produced that checks a proposed parameter set with EXPLICIT, parameter-naming error messages (per the user's requirement).

**Files:**
- Create: `include/goss/model/parameter.hpp` (`ParameterHandle`, `ParameterSpec`, `ParameterValidator`)
- Modify: `include/goss/model/handles.hpp:1-25` (add `ParameterHandle`)
- Modify: `include/goss/model/model.hpp` (add `add_parameter`, storage, and validator construction)
- Modify: `include/goss/transcription/ocp_problem.hpp:21-36` (carry parameter count + defaults)
- Test: `tests/model/test_parameters.cpp`
- Modify: `CMakeLists.txt:155-164`

**Design note (flagged deviation from the user's phrasing "compiled artifact"):** parameter validation is branchy, message-heavy control flow whose value is *readable, specific errors* — not float throughput. Per the org standard on flagging deviations: I implement the validator as a plain C++ `ParameterValidator` object bundled with the compiled problem (produced at build/compile time), NOT as JIT-generated C. It is still a build-time artifact tied to the compiled model; it simply is not machine-generated code. Justification: correctness/clarity of messages and zero JIT cost for a cheap check. If genuinely JIT'd validation is later required, it can be added behind the same `validate()` interface without changing callers.

**Interfaces:**
- Produces:
  - `struct model::ParameterHandle { std::size_t index; operator std::size_t() const; };`
  - `struct model::ParameterSpec { std::string name; double default_value; double lower_bound; double upper_bound; };`
  - `class model::ParameterValidator` with:
    ```cpp
    explicit ParameterValidator(std::vector<ParameterSpec> specs);
    std::size_t size() const;
    const std::vector<double>& defaults() const;   // default_value per spec, in order
    void validate(const std::vector<double>& values) const;  // throws ModelError with explicit message
    ```
  - `model::ParameterHandle Model::add_parameter(const std::string& name, double default_value, double lower_bound = -kInf, double upper_bound = kInf);`
  - `std::size_t Model::num_parameters() const;`
  - `model::ParameterValidator Model::parameter_validator() const;`
  - `OcpProblem` gains: `std::size_t num_parameters; std::vector<double> parameter_defaults;`

- [ ] **Step 1: Write the failing validator test (explicit messages are the contract)**

```cpp
#include <gtest/gtest.h>
#include <string>
#include "goss/model/model.hpp"
#include "goss/model/parameter.hpp"
#include "goss/model/errors.hpp"

TEST(ModelParameters, ValidatorAcceptsInRangeAndReportsDefaults) {
    goss::model::Model model;
    model.add_parameter("arrival_rate", /*default=*/2.0, /*lower=*/0.0, /*upper=*/10.0);
    model.add_parameter("cost_weight",  /*default=*/1.0, /*lower=*/0.0, /*upper=*/100.0);

    ASSERT_EQ(model.num_parameters(), 2u);
    auto validator = model.parameter_validator();
    ASSERT_EQ(validator.size(), 2u);
    EXPECT_EQ(validator.defaults(), (std::vector<double>{2.0, 1.0}));
    EXPECT_NO_THROW(validator.validate({3.0, 50.0}));
}

TEST(ModelParameters, ValidatorRejectsWrongSizeWithExplicitMessage) {
    goss::model::Model model;
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    auto validator = model.parameter_validator();
    try {
        validator.validate({1.0, 2.0});
        FAIL() << "expected ModelError";
    } catch (const goss::model::ModelError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("expected 1"), std::string::npos);
        EXPECT_NE(message.find("got 2"), std::string::npos);
    }
}

TEST(ModelParameters, ValidatorRejectsOutOfRangeNamingTheParameter) {
    goss::model::Model model;
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    auto validator = model.parameter_validator();
    try {
        validator.validate({99.0});
        FAIL() << "expected ModelError";
    } catch (const goss::model::ModelError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("arrival_rate"), std::string::npos);  // names the offender
        EXPECT_NE(message.find("99"), std::string::npos);
        EXPECT_NE(message.find("10"), std::string::npos);            // reports the bound
    }
}
```

- [ ] **Step 2: Wire test TU; run to verify FAIL**

Add `tests/model/test_parameters.cpp` to `goss_model_tests` (`CMakeLists.txt:155-164`).

Run: `cmake --build build --target goss_model_tests`
Expected: FAIL — `parameter.hpp` / `add_parameter` missing.

- [ ] **Step 3: Create `parameter.hpp`**

```cpp
// include/goss/model/parameter.hpp
#pragma once
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include "goss/model/errors.hpp"

namespace goss::model {

struct ParameterHandle {
    std::size_t index;
    constexpr operator std::size_t() const noexcept { return index; }
};

struct ParameterSpec {
    std::string name;
    double default_value;
    double lower_bound;
    double upper_bound;
};

/// Build-time artifact bundled with a compiled problem. Validates a proposed
/// parameter set against declared size and per-parameter bounds, throwing
/// ModelError with a message that names the offending parameter and its bound.
class ParameterValidator {
 public:
    explicit ParameterValidator(std::vector<ParameterSpec> specs)
        : specs_(std::move(specs)) {
        defaults_.reserve(specs_.size());
        for (const ParameterSpec& spec : specs_) defaults_.push_back(spec.default_value);
    }

    std::size_t size() const { return specs_.size(); }
    const std::vector<double>& defaults() const { return defaults_; }

    void validate(const std::vector<double>& values) const {
        if (values.size() != specs_.size())
            throw ModelError("ParameterValidator: expected " +
                             std::to_string(specs_.size()) + " parameter(s), got " +
                             std::to_string(values.size()));
        for (std::size_t i = 0; i < specs_.size(); ++i) {
            const ParameterSpec& spec = specs_[i];
            if (std::isnan(values[i]))
                throw ModelError("ParameterValidator: parameter '" + spec.name +
                                 "' is NaN");
            if (values[i] < spec.lower_bound)
                throw ModelError("ParameterValidator: parameter '" + spec.name +
                                 "' value " + std::to_string(values[i]) +
                                 " is below its lower bound " +
                                 std::to_string(spec.lower_bound));
            if (values[i] > spec.upper_bound)
                throw ModelError("ParameterValidator: parameter '" + spec.name +
                                 "' value " + std::to_string(values[i]) +
                                 " exceeds its upper bound " +
                                 std::to_string(spec.upper_bound));
        }
    }

 private:
    std::vector<ParameterSpec> specs_;
    std::vector<double> defaults_;
};

}  // namespace goss::model
```

- [ ] **Step 4: Add parameter storage + API to `Model`**

In `model.hpp`, include `"goss/model/parameter.hpp"`, add private members and methods:

```cpp
    ParameterHandle add_parameter(const std::string& name, double default_value,
                                  double lower_bound = -transcription::kInf,
                                  double upper_bound =  transcription::kInf) {
        ensure_unique_name(name);
        if (lower_bound > upper_bound)
            throw ModelError("add_parameter: lower > upper for parameter '" + name + "'");
        if (default_value < lower_bound || default_value > upper_bound)
            throw ModelError("add_parameter: default for parameter '" + name +
                             "' is outside its bounds");
        const std::size_t index = parameter_specs_.size();
        parameter_specs_.push_back(ParameterSpec{name, default_value, lower_bound, upper_bound});
        return ParameterHandle{index};
    }

    std::size_t num_parameters() const { return parameter_specs_.size(); }
    ParameterValidator parameter_validator() const { return ParameterValidator(parameter_specs_); }
```

Add `std::vector<ParameterSpec> parameter_specs_;` to the private section. Extend `ensure_unique_name` to also scan `parameter_specs_` (so a parameter can't collide with a state/control name and vice-versa).

- [ ] **Step 5: Carry parameter metadata into `OcpProblem`**

In `ocp_problem.hpp`, add to the struct (`ocp_problem.hpp:22-36`):

```cpp
    std::size_t num_parameters = 0;
    std::vector<double> parameter_defaults;   // size == num_parameters
```

In `Model::build` (`model.hpp:158-172`), append to the aggregate initializer:

```cpp
        /* num_parameters       */ parameter_specs_.size(),
        /* parameter_defaults   */ [this]{
            std::vector<double> defaults;
            defaults.reserve(parameter_specs_.size());
            for (const ParameterSpec& spec : parameter_specs_) defaults.push_back(spec.default_value);
            return defaults;
        }()
```

- [ ] **Step 6: Run tests to verify PASS**

Run: `cd build && ctest -R ModelParameters --output-on-failure`
Expected: PASS.

- [ ] **Step 7: Guard against name collision (regression test) + run full model suite**

```cpp
TEST(ModelParameters, ParameterNameCollidesWithStateThrows) {
    goss::model::Model model;
    model.add_state("q");
    EXPECT_THROW(model.add_parameter("q", 1.0), goss::model::ModelError);
}
```

Run: `cd build && ctest -R goss_model_tests --output-on-failure` (ensure existing model tests still pass — additive change).
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add include/goss/model/parameter.hpp include/goss/model/model.hpp include/goss/transcription/ocp_problem.hpp tests/model/test_parameters.cpp CMakeLists.txt
git commit -m "feat(model): declare parameters + build-time ParameterValidator with explicit messages"
```

---

### Task 5: `transcription/` — thread parameters into the packed functors

Teach the packed functors in all schemes to call dynamics/cost with the parameter vector *when the functor accepts it* (opt-in 4-argument overload), and to record the backend with the parameter count. Existing 3-argument functors continue to work via compile-time dispatch — additive, no rewrite.

**Files:**
- Create: `include/goss/transcription/invoke.hpp` (arity-detecting call helpers)
- Modify: `include/goss/transcription/trapezoidal.hpp:45-93`
- Modify: `include/goss/transcription/hermite_simpson.hpp` (same packed-functor edits)
- Modify: `include/goss/transcription/legendre_gauss_lobatto.hpp` (same)
- Test: `tests/transcription/test_parametric_transcription.cpp`
- Modify: `CMakeLists.txt:133-148`

**Interfaces:**
- Consumes: `OcpProblem::num_parameters`, `OcpProblem::parameter_defaults` (Task 4); `CppADCGBackend` parameterized constructor (Task 2).
- Produces:
  - `goss::transcription::detail::call_dynamics(dynamics, x, u, p, t)` — calls `dynamics(x,u,p,t)` if invocable, else `dynamics(x,u,t)`.
  - `goss::transcription::detail::call_cost(cost, x, u, p, t)` — same rule for cost.
  - Each scheme's `compile` now builds a `(z,p)` packed functor and constructs `CppADCGBackend` with `ocp.num_parameters` / `ocp.parameter_defaults` when `num_parameters > 0`; otherwise uses the existing single-argument path unchanged.

- [ ] **Step 1: Write the arity-dispatch helper (unit test first)**

```cpp
// tests/transcription/test_parametric_transcription.cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/transcription/invoke.hpp"

namespace {
struct ThreeArgDyn {  // legacy: (x, u, t)
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>&, T) const {
        return { -x[0] };
    }
};
struct FourArgDyn {   // parametric: (x, u, p, t)
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>&,
                              const std::vector<T>& p, T) const {
        return { p[0] - x[0] };  // arrival_rate - x
    }
};
}  // namespace

TEST(TranscriptionInvoke, DispatchesToLegacyThreeArg) {
    std::vector<double> x{5.0}, u{}, p{2.0};
    auto out = goss::transcription::detail::call_dynamics(ThreeArgDyn{}, x, u, p, 0.0);
    EXPECT_DOUBLE_EQ(out[0], -5.0);   // ignores p
}

TEST(TranscriptionInvoke, DispatchesToParametricFourArg) {
    std::vector<double> x{5.0}, u{}, p{2.0};
    auto out = goss::transcription::detail::call_dynamics(FourArgDyn{}, x, u, p, 0.0);
    EXPECT_DOUBLE_EQ(out[0], 2.0 - 5.0);  // uses p
}
```

- [ ] **Step 2: Wire test TU; run to verify FAIL**

Add `tests/transcription/test_parametric_transcription.cpp` to `goss_transcription_tests` (`CMakeLists.txt:133-141`).

Run: `cmake --build build --target goss_transcription_tests`
Expected: FAIL — `invoke.hpp` missing.

- [ ] **Step 3: Implement `invoke.hpp`**

```cpp
// include/goss/transcription/invoke.hpp
#pragma once
#include <type_traits>
#include <utility>
#include <vector>

namespace goss::transcription::detail {

// Calls dynamics(x,u,p,t) when that overload exists, otherwise dynamics(x,u,t).
template <typename Fn, typename T>
auto call_dynamics(const Fn& fn, const std::vector<T>& x, const std::vector<T>& u,
                   const std::vector<T>& p, T t)
    -> std::vector<T> {
    if constexpr (std::is_invocable_v<const Fn&, const std::vector<T>&,
                                      const std::vector<T>&, const std::vector<T>&, T>) {
        return fn(x, u, p, t);
    } else {
        (void)p;
        return fn(x, u, t);
    }
}

template <typename Fn, typename T>
auto call_cost(const Fn& fn, const std::vector<T>& x, const std::vector<T>& u,
               const std::vector<T>& p, T t) -> T {
    if constexpr (std::is_invocable_v<const Fn&, const std::vector<T>&,
                                      const std::vector<T>&, const std::vector<T>&, T>) {
        return fn(x, u, p, t);
    } else {
        (void)p;
        return fn(x, u, t);
    }
}

}  // namespace goss::transcription::detail
```

Run: `cd build && ctest -R TranscriptionInvoke --output-on-failure`
Expected: PASS.

- [ ] **Step 4: Thread parameters through the Trapezoidal packed functor**

In `trapezoidal.hpp`, include `"goss/transcription/invoke.hpp"`. Change the packed lambda to capture and accept `p`:

```cpp
    const std::size_t np = ocp.num_parameters;
    auto packed = [ocp, layout, ns, nc, nn, ni, node_times, np]
                  (const auto& z, const auto& p) {
        using T = typename std::decay_t<decltype(z)>::value_type;
        // ... state_at / control_at unchanged ...
        // cost term:
        T Lk  = detail::call_cost(ocp.cost, state_at(k),     control_at(k),     p, tk);
        T Lk1 = detail::call_cost(ocp.cost, state_at(k + 1), control_at(k + 1), p, tk1);
        // defects:
        auto fk  = detail::call_dynamics(ocp.dynamics, xk,  control_at(k),     p, tk);
        auto fk1 = detail::call_dynamics(ocp.dynamics, xk1, control_at(k + 1), p, tk1);
        // ... rest unchanged ...
    };
```

Construct the backend with the parameterized constructor when there are parameters:

```cpp
    std::unique_ptr<goss::ad::CppADCGBackend> backend;
    if (np > 0) {
        backend = std::make_unique<goss::ad::CppADCGBackend>(
            packed, layout.total_variables(), np, ocp.parameter_defaults, model_name);
    } else {
        // Wrap the two-arg packed functor as one-arg (empty p) to reuse the
        // existing single-argument constructor path unchanged.
        auto packed_no_params = [packed](const auto& z) {
            using T = typename std::decay_t<decltype(z)>::value_type;
            return packed(z, std::vector<T>{});
        };
        backend = std::make_unique<goss::ad::CppADCGBackend>(
            packed_no_params, layout.total_variables(), model_name);
    }
```

The rest of `compile` (bounds, defect constraint bounds, `NLPProblem` construction) is unchanged.

- [ ] **Step 5: Repeat the identical packed-functor edits for Hermite-Simpson and LGL**

Apply the same three edits (capture `np`+`p`, use `call_dynamics`/`call_cost`, branch backend construction) to `hermite_simpson.hpp` and `legendre_gauss_lobatto.hpp`. The packed-functor bodies differ per scheme but the parameter-threading edits are mechanical and identical in shape.

- [ ] **Step 6: Add a parametric-compile-and-eval integration test**

```cpp
#include "goss/model/model.hpp"
#include "goss/transcription/trapezoidal.hpp"

TEST(ParametricTranscription, CompilesOnceAndTracksParameterAcrossEvals) {
    goss::model::Model model;
    auto x = model.add_state("x");
    auto rate = model.add_parameter("arrival_rate", /*default=*/1.0, 0.0, 10.0);
    (void)rate;
    model.set_initial_state(x, 0.0);
    model.set_mesh(0.0, 1.0, 5);

    // dx/dt = arrival_rate - x  (uses the 4-arg parametric overload)
    auto dynamics = [](const auto& xx, const auto&, const auto& p, auto t) {
        using T = std::decay_t<decltype(t)>;
        return std::vector<T>{ p[0] - xx[0] };
    };
    auto cost = [](const auto&, const auto&, const auto&, auto t) {
        using T = std::decay_t<decltype(t)>; return T(0);
    };
    auto ocp = model.build(dynamics, cost);
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "param_trap");

    ASSERT_EQ(compiled.problem->num_parameters(), 1u);
    // Objective is zero here; assert a defect constraint value shifts with p.
    std::vector<double> z(compiled.layout.total_variables(), 0.0);
    compiled.problem->set_parameters({1.0});
    auto g1 = compiled.problem->eval_constraints(z);
    compiled.problem->set_parameters({4.0});
    auto g2 = compiled.problem->eval_constraints(z);
    // First defect: x1 - x0 - (h/2)(f0+f1); at z=0, f = p - 0 = p, so g = -(h)*p.
    EXPECT_NE(g1.front(), g2.front());
}
```

Run: `cd build && ctest -R "ParametricTranscription|TranscriptionInvoke" --output-on-failure`
Expected: PASS.

- [ ] **Step 7: Run the full transcription + accuracy suites (additive-safety check)**

Run: `cmake --build build && cd build && ctest -R "goss_transcription_tests|goss_accuracy_tests" --output-on-failure`
Expected: PASS — existing 3-arg models unaffected by the arity dispatch.

- [ ] **Step 8: Commit**

```bash
git add include/goss/transcription/invoke.hpp include/goss/transcription/trapezoidal.hpp include/goss/transcription/hermite_simpson.hpp include/goss/transcription/legendre_gauss_lobatto.hpp tests/transcription/test_parametric_transcription.cpp CMakeLists.txt
git commit -m "feat(transcription): thread optional parameters through packed functors (arity-dispatched)"
```

---

### Task 6: `sim/` — validate-then-bind helper + CompiledOcp validator handoff

Bundle the `ParameterValidator` with the compiled problem so a caller has one place to validate a parameter set (explicit errors) before injecting it. Provide a `sim::apply_parameters` helper that validates then sets — the single entry point the sweep harness (Plan B) will call.

**Files:**
- Modify: `include/goss/transcription/transcription.hpp:22-25` (add optional validator to `CompiledOcp`)
- Create: `include/goss/sim/parameters.hpp` (`apply_parameters`)
- Modify: each scheme `compile` to attach the validator built from `ocp` metadata
- Test: `tests/sim/test_parameters.cpp`
- Modify: `CMakeLists.txt:177-188`

**Interfaces:**
- Consumes: `nlp::NLPProblem::set_parameters` (Task 3), `model::ParameterValidator` (Task 4).
- Produces:
  - `CompiledOcp` gains `model::ParameterValidator validator;` (constructed from `ocp.num_parameters` + `ocp.parameter_defaults`; when zero params, an empty validator).
  - `void sim::apply_parameters(nlp::NLPProblem& problem, const model::ParameterValidator& validator, const std::vector<double>& values)` — calls `validator.validate(values)` then `problem.set_parameters(values)`; propagates `ModelError` (validation) or `ADError` (injection).

**Design note:** `CompiledOcp` currently holds only `problem` + `layout` (`transcription.hpp:22-25`). Adding the validator keeps parameter metadata co-located with the compiled model, so a sweep worker holds one self-contained object. Since `ParameterValidator` needs names for good messages and `OcpProblem` currently carries only defaults, extend `OcpProblem` (Task 4) minimally OR pass names through: to keep messages naming parameters, carry `std::vector<std::string> parameter_names` in `OcpProblem` too (add in this task if not already present) and build `ParameterSpec`s from names+defaults+bounds. If bounds aren't threaded into `OcpProblem`, default them to `[-kInf, kInf]` and rely on the model-level validator obtained via `Model::parameter_validator()` for bound checks. **Simplest correct choice:** the authoritative validator is `Model::parameter_validator()`; `CompiledOcp.validator` is a copy of it. Have each scheme's `compile` accept the validator — but `compile` takes `OcpProblem`, not `Model`. Therefore thread `parameter_names` + `parameter_lower`/`parameter_upper` into `OcpProblem` in Task 4's Step 5 (extend the added fields to include names and bounds), and build the validator inside `compile` from those. Update Task 4 Step 5 accordingly if implementing tasks strictly in order.

- [ ] **Step 1: Extend OcpProblem parameter metadata (names + bounds)**

Ensure `ocp_problem.hpp` carries (superset of Task 4's fields):

```cpp
    std::size_t num_parameters = 0;
    std::vector<std::string> parameter_names;    // size == num_parameters
    std::vector<double> parameter_defaults;      // size == num_parameters
    std::vector<double> parameter_lower;         // size == num_parameters
    std::vector<double> parameter_upper;         // size == num_parameters
```

Update `Model::build` to populate all four from `parameter_specs_`. (Add `#include <string>` to `ocp_problem.hpp`.)

- [ ] **Step 2: Write the failing sim test**

```cpp
// tests/sim/test_parameters.cpp
#include <gtest/gtest.h>
#include <string>
#include "goss/model/model.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/sim/parameters.hpp"
#include "goss/model/errors.hpp"

TEST(SimParameters, ValidateThenBindAcceptsInRange) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.add_parameter("arrival_rate", 1.0, 0.0, 10.0);
    model.set_initial_state(x, 0.0);
    model.set_mesh(0.0, 1.0, 4);
    auto ocp = model.build(
        [](const auto& xx, const auto&, const auto& p, auto){ using T=std::decay_t<decltype(xx[0])>; return std::vector<T>{p[0]-xx[0]}; },
        [](const auto&, const auto&, const auto&, auto t){ using T=std::decay_t<decltype(t)>; return T(0); });
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "sim_param_ok");

    ASSERT_EQ(compiled.validator.size(), 1u);
    EXPECT_NO_THROW(goss::sim::apply_parameters(*compiled.problem, compiled.validator, {3.0}));
}

TEST(SimParameters, ValidateThenBindRejectsOutOfRangeBeforeTouchingSolver) {
    goss::model::Model model;
    auto x = model.add_state("x");
    model.add_parameter("arrival_rate", 1.0, 0.0, 10.0);
    model.set_initial_state(x, 0.0);
    model.set_mesh(0.0, 1.0, 4);
    auto ocp = model.build(
        [](const auto& xx, const auto&, const auto& p, auto){ using T=std::decay_t<decltype(xx[0])>; return std::vector<T>{p[0]-xx[0]}; },
        [](const auto&, const auto&, const auto&, auto t){ using T=std::decay_t<decltype(t)>; return T(0); });
    auto compiled = goss::transcription::Trapezoidal::compile(ocp, "sim_param_bad");

    try {
        goss::sim::apply_parameters(*compiled.problem, compiled.validator, {50.0});
        FAIL() << "expected ModelError";
    } catch (const goss::model::ModelError& error) {
        EXPECT_NE(std::string(error.what()).find("arrival_rate"), std::string::npos);
    }
}
```

- [ ] **Step 3: Wire test TU; run to verify FAIL**

Add `tests/sim/test_parameters.cpp` to `goss_sim_tests` (`CMakeLists.txt:177-182`).

Run: `cmake --build build --target goss_sim_tests`
Expected: FAIL — `CompiledOcp.validator` and `sim/parameters.hpp` missing.

- [ ] **Step 4: Add validator to CompiledOcp**

In `transcription.hpp`, include `"goss/model/parameter.hpp"` and extend:

```cpp
struct CompiledOcp {
    std::unique_ptr<nlp::NLPProblem> problem;
    VariableLayout layout;
    model::ParameterValidator validator{ {} };   // empty by default (no params)
};
```

(Guard against an include cycle: `parameter.hpp` depends only on `model/errors.hpp`, not on transcription — safe.)

- [ ] **Step 5: Build the validator inside each scheme `compile`**

In each scheme's `compile`, after building `problem`, construct the validator from the `OcpProblem` parameter metadata and return it:

```cpp
    std::vector<model::ParameterSpec> specs;
    specs.reserve(ocp.num_parameters);
    for (std::size_t i = 0; i < ocp.num_parameters; ++i)
        specs.push_back(model::ParameterSpec{
            ocp.parameter_names[i], ocp.parameter_defaults[i],
            ocp.parameter_lower[i], ocp.parameter_upper[i]});
    return CompiledOcp{std::move(problem), layout, model::ParameterValidator(std::move(specs))};
```

- [ ] **Step 6: Implement `sim/parameters.hpp`**

```cpp
// include/goss/sim/parameters.hpp
#pragma once
#include <vector>
#include "goss/model/parameter.hpp"
#include "goss/nlp/nlp_problem.hpp"

namespace goss::sim {

/// Validates a proposed parameter set against the compiled problem's validator
/// (throwing ModelError with an explicit, parameter-naming message on failure),
/// then injects it into the NLP (may throw ADError on a backend size mismatch).
/// This is the single entry point a sweep runner uses per parameter point.
inline void apply_parameters(nlp::NLPProblem& problem,
                             const model::ParameterValidator& validator,
                             const std::vector<double>& values) {
    validator.validate(values);       // explicit errors, before any solver work
    problem.set_parameters(values);   // compile-once injection
}

}  // namespace goss::sim
```

- [ ] **Step 7: Run tests to verify PASS**

Run: `cd build && ctest -R SimParameters --output-on-failure`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add include/goss/transcription/transcription.hpp include/goss/transcription/ocp_problem.hpp include/goss/model/model.hpp include/goss/sim/parameters.hpp include/goss/transcription/trapezoidal.hpp include/goss/transcription/hermite_simpson.hpp include/goss/transcription/legendre_gauss_lobatto.hpp tests/sim/test_parameters.cpp CMakeLists.txt
git commit -m "feat(sim): bundle ParameterValidator with CompiledOcp; add validate-then-bind helper"
```

---

### Task 7: End-to-end — parametric queue solved twice, compiled once

The spec's motivating example (the queue) becomes a permanent parametric fixture: declare `arrival_rate` as a parameter, compile once, solve at two arrival rates, and assert (a) both solves succeed, (b) the optima differ, (c) only ONE compilation occurred.

**Files:**
- Test: `tests/accuracy/test_parametric_queue.cpp`
- Modify: `CMakeLists.txt:215-219` (add to `goss_accuracy_tests`)

**Interfaces:**
- Consumes: everything above — `Model::add_parameter`, `Trapezoidal::compile`, `sim::apply_parameters`, `IpoptSolver::solve`, `sim::linear_guess`.

- [ ] **Step 1: Write the end-to-end test**

```cpp
// tests/accuracy/test_parametric_queue.cpp
#include <gtest/gtest.h>
#include <vector>
#include "goss/model/model.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/sim/initial_guess.hpp"
#include "goss/sim/parameters.hpp"

TEST(ParametricQueue, CompileOnceSolveManyAcrossArrivalRates) {
    constexpr double MAX_RATE = 5.0;
    goss::model::Model model;
    auto q    = model.add_state("queue_length");
    auto rate = model.add_control("service_rate");
    auto arrival = model.add_parameter("arrival_rate", /*default=*/2.0, 0.0, 10.0);
    (void)arrival;
    model.set_state_bounds(q, 0.0, 1e19);            // q >= 0
    model.set_control_bounds(rate, 0.0, MAX_RATE);
    model.set_initial_state(q, 10.0);
    model.set_mesh(0.0, 5.0, 30);

    // dq/dt = arrival_rate - service_rate ; cost = integral(q + 0.1*rate^2)
    auto dynamics = [](const auto& x, const auto& u, const auto& p, auto) {
        using T = std::decay_t<decltype(x[0])>;
        return std::vector<T>{ p[0] - u[0] };
    };
    auto cost = [](const auto& x, const auto& u, const auto&, auto) {
        using T = std::decay_t<decltype(x[0])>;
        return x[0] + T(0.1) * u[0] * u[0];
    };

    auto ocp = model.build(dynamics, cost);
    // COMPILE ONCE.
    auto compiled = goss::transcription::HermiteSimpson::compile(ocp, "parametric_queue");
    const auto z0 = goss::sim::linear_guess(model, compiled.layout);

    goss::solver::IpoptSolver solver;

    goss::sim::apply_parameters(*compiled.problem, compiled.validator, {1.0});
    auto low = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(low.status, goss::solver::SolverStatus::Success);

    goss::sim::apply_parameters(*compiled.problem, compiled.validator, {4.0});
    auto high = solver.solve(*compiled.problem, z0);
    ASSERT_EQ(high.status, goss::solver::SolverStatus::Success);

    // Higher arrival rate => costlier optimum (queue harder to drain).
    EXPECT_GT(high.objective_value, low.objective_value);
}
```

- [ ] **Step 2: Wire test TU; run to verify FAIL then PASS**

Add `tests/accuracy/test_parametric_queue.cpp` to `goss_accuracy_tests` (`CMakeLists.txt:215-219`).

Run: `cmake --build build --target goss_accuracy_tests && cd build && ctest -R ParametricQueue --output-on-failure`
Expected: PASS (all prior tasks in place).

- [ ] **Step 3: Prove compile-once with a JIT-artifact count assertion (optional hardening)**

The generated `.so` for `parametric_queue` should appear exactly once under `build/`. Add a check that a second `apply_parameters`+`solve` did not create a new `parametric_queue*.so`:

```cpp
// (In the same test, before/after the two solves, or as a sibling test using
//  std::filesystem to count files in the build dir matching "parametric_queue".)
```

If filesystem inspection is too brittle across environments, rely instead on wall-clock: assert the second solve's setup path spent no time compiling by timing `apply_parameters` (must be sub-millisecond). Keep whichever is stable in CI; document the choice inline.

Run: `cd build && ctest -R ParametricQueue --output-on-failure`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/accuracy/test_parametric_queue.cpp CMakeLists.txt
git commit -m "test(accuracy): parametric queue compiles once, solves across arrival rates"
```

---

## Self-Review

**Spec/requirement coverage:**
- Compile-once parameter injection → Tasks 1–3 (mechanism, backend, NLP).
- Full DSL threading (user's chosen scope) → Tasks 4–5 (Model declares parameters; schemes thread them).
- Parameter-validation artifact with explicit messages (user's explicit requirement) → Task 4 (`ParameterValidator`) + Task 6 (`CompiledOcp.validator`, `sim::apply_parameters`).
- "Each process picks up new params, no recompile" (user note) → guaranteed by Tasks 2–3 (`set_parameters` on a compiled model) and demonstrated in Task 7; consumed by Plan B.
- Additive/no-rewrite invariant → Task 5 arity dispatch keeps all existing 3-arg models green (verified in Task 5 Step 7).

**Placeholder scan:** No TBD/TODO. The one genuine unknown (CppADCodeGen dynamic-parameter API) is isolated in Task 1 as a spike with a concrete fallback, not left as a placeholder downstream.

**Type consistency:** `set_parameters(const std::vector<double>&)`, `num_parameters()`, `ParameterValidator::validate/size/defaults`, `CompiledOcp.validator`, and `sim::apply_parameters` signatures are used identically across Tasks 2–7. `OcpProblem` parameter fields (`num_parameters`, `parameter_names`, `parameter_defaults`, `parameter_lower`, `parameter_upper`) are introduced in Task 4/6 and consumed consistently in Task 5/6.

**Cross-plan handoff:** Plan B (parallel sweep) depends on `sim::apply_parameters`, `CompiledOcp.validator`, and compile-once `set_parameters` delivered here.
