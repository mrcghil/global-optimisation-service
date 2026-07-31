# Solver Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `solver/` layer — a backend-agnostic `Solver` interface plus two concrete adapters (IPOPT via its `TNLP` interface, and a derivative-free NLopt/COBYLA baseline) that solve a `goss::nlp::NLPProblem`, validated against a hand-ported Hock-Schittkowski subset with known optima.

**Architecture:** `NLPProblem`'s public API is a near 1:1 map onto IPOPT's `TNLP`: `eval_objective`→`eval_f`, `eval_objective_gradient`→`eval_grad_f`, `eval_constraints`→`eval_g`, `constraint_jacobian_sparsity`+`eval_constraint_jacobian`→`eval_jac_g` (two-call), `lagrangian_hessian_sparsity`+`eval_lagrangian_hessian(x,σ,λ)`→`eval_h` (two-call, lower-triangle). So `IpoptTNLPAdapter` is mostly pure translation. `NloptSolver` (COBYLA) feeds the same `NLPProblem` methods through NLopt's callback form, decomposing two-sided constraints. Both implement one abstract `Solver` returning a `SolverResult`. The NLopt baseline lets us cross-check IPOPT independently of the derivative path.

**Tech Stack:** C++17, the merged `goss::ad` + `goss::nlp` layers, IPOPT 3.11.9 (`coinor-libipopt-dev`, via pkg-config), NLopt 2.7.1 (`libnlopt-cxx-dev`, via pkg-config), GoogleTest, CMake, containerized build via `scripts/dev.sh`.

## Global Constraints

- Language: **C++17**.
- The `Solver` interface MUST depend only on `goss::nlp::NLPProblem` + `goss::nlp` types — NEVER on IPOPT or NLopt types. Solver-neutrality is load-bearing (mirrors how `NLPProblem` is backend-agnostic).
- IPOPT- and NLopt-specific types appear ONLY inside their respective adapter `.cpp` files (`ipopt_solver.cpp`, `nlopt_solver.cpp`), never in any public header.
- IPOPT version is **3.11.9** (Ubuntu `coinor-libipopt-dev`). It predates 3.14, so do NOT reference `Maximum_WallTime_Exceeded` (added in 3.14). Autotools build ships NO CMake config — discover via **pkg-config** (`ipopt`). **Verified in Task 1:** headers live under `/usr/include/coin/`, so the include form is `<IpTNLP.hpp>` / `<IpIpoptApplication.hpp>` (NO `coin-or/` prefix), and IPOPT 3.11.9 headers require `-DHAVE_CSTDDEF` — both supplied by the `goss_ipopt_iface` INTERFACE target created in Task 1 (link it PRIVATE to inherit the include dir + flag). `liblapack-dev`/`libblas-dev` are in the image for IPOPT's link deps.
- NLopt is discovered via **pkg-config** (`nlopt`); C++ header `<nlopt.hpp>`; both `libnlopt-dev` and `libnlopt-cxx-dev` installed. Constraint sign convention: `fc(x) <= 0`.
- `Ipopt::TNLP` subclasses MUST be heap-allocated and held by `Ipopt::SmartPtr` — never `delete`d manually. Index style is **C_STYLE (0-based)**, matching `goss`'s sparsity patterns. `Index`=int, `Number`=double.
- IPOPT `eval_h` computes `σ·∇²f + Σλᵢ∇²gᵢ` (lower-triangle) — this is EXACTLY `NLPProblem::eval_lagrangian_hessian(x, σ, λ)`. Use exact Hessian (`hessian_approximation=exact`), not limited-memory.
- Error handling per org standard: a `SolverError` exception for setup/usage errors with meaningful messages; solve outcomes (including non-convergence) are reported via `SolverResult.status`, NOT exceptions. No silent catch-all.
- NLopt throws on failure by default — the adapter MUST call `set_exceptions_enabled(false)` and inspect the return code, OR catch NLopt exceptions and translate to `SolverResult.status`. Do not let NLopt exceptions escape `solve()`.
- Coding standards: verbose descriptive variable names; type annotations everywhere; comments explain WHY.
- Container-first build: all `cmake`/`ctest` run inside the container via `scripts/dev.sh '<command>'`. Never build on the host.
- Test framework: **GoogleTest**. HS fixtures are packed functors (objective in output 0, constraints in outputs 1..m) fed to a real `CppADCGBackend` → `NLPProblem`.
- HS optima are test oracles: assert objective within ~1e-4 (looser for NLopt/COBYLA, ~1e-2). HS76 is EXCLUDED (uncertain published x*). HS35 is a QP (NOT "Beale").

---

## File Structure

- `include/goss/solver/errors.hpp` — `SolverError : std::runtime_error`.
- `include/goss/solver/solver_result.hpp` — `SolverStatus` enum + `SolverResult` struct (solution, objective, multipliers, message).
- `include/goss/solver/solver.hpp` — abstract `Solver` interface.
- `include/goss/solver/ipopt_solver.hpp` — `IpoptSolver : Solver` declaration (no IPOPT types in the header).
- `src/solver/ipopt_solver.cpp` — `IpoptSolver` + the internal `IpoptTNLPAdapter` (all IPOPT types confined here).
- `include/goss/solver/nlopt_solver.hpp` — `NloptSolver : Solver` declaration (no NLopt types in the header).
- `src/solver/nlopt_solver.cpp` — `NloptSolver` implementation (all NLopt types confined here).
- `tests/solver/hs_fixtures.hpp` — packed HS problem functors + a helper to build the matching `NLPProblem` (bounds).
- `tests/solver/test_solver_interface.cpp` — interface/result contract tests.
- `tests/solver/test_ipopt_solver.cpp` — IPOPT solves (QP smoke, then HS problems).
- `tests/solver/test_nlopt_solver.cpp` — NLopt solves.
- `tests/solver/test_solver_agreement.cpp` — IPOPT vs NLopt cross-check.
- `Dockerfile` — add IPOPT + NLopt apt packages to the `build-base` stage.
- `CMakeLists.txt` — pkg-config discovery of ipopt/nlopt; `goss_solver` lib + `goss_solver_tests`.

---

### Task 1: Add IPOPT + NLopt to the container and wire pkg-config discovery

**Files:**
- Modify: `Dockerfile`
- Modify: `CMakeLists.txt`
- Create: `tests/solver/test_solver_link.cpp`

**Interfaces:**
- Consumes: nothing new.
- Produces: a rebuilt container image with IPOPT 3.11.9 + NLopt 2.7.1 dev packages; CMake `IPOPT`/`NLOPT` imported flags via pkg-config; a `goss_solver_tests` executable; a link test proving both headers include and both libs link.

**Research facts (verified):** apt packages are `coinor-libipopt-dev` (3.11.9) and `libnlopt-dev` + `libnlopt-cxx-dev` (2.7.1). Both ship pkg-config files named `ipopt` and `nlopt`. `pkg-config` is already in the image. IPOPT headers install under `/usr/include/coin-or/`.

- [ ] **Step 1: Add the apt packages to the Dockerfile**

In `Dockerfile`, find the `apt-get install` line in the `build-base` stage (it currently installs `g++ gcc make cmake git ca-certificates libeigen3-dev`). Add the three solver packages:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ gcc make cmake git ca-certificates libeigen3-dev pkg-config \
        coinor-libipopt-dev libnlopt-dev libnlopt-cxx-dev \
    && rm -rf /var/lib/apt/lists/*
```
(Keep the existing packages; add `pkg-config` explicitly too even though it is present, so the image is self-describing. Preserve the Zscaler CA lines and everything else in the Dockerfile unchanged.)

- [ ] **Step 2: Rebuild the container image**

Run: `docker build --target build-base -t goss-build-base:local --build-arg INJECT_ZSCALER_CA=true .`
Expected: image builds; the apt layer now includes ipopt + nlopt. (First rebuild is slow.)
Then verify the deps: `scripts/dev.sh 'pkg-config --exists ipopt && pkg-config --exists nlopt && echo PKGCONFIG_OK; ls /usr/include/coin-or/IpTNLP.hpp; ls /usr/include/nlopt.hpp'`
Expected: prints `PKGCONFIG_OK` and both header paths.

- [ ] **Step 3: Write the failing link test**

```cpp
// tests/solver/test_solver_link.cpp
#include <gtest/gtest.h>
#include <coin-or/IpIpoptApplication.hpp>
#include <coin-or/IpTNLP.hpp>
#include <nlopt.hpp>

TEST(SolverLink, IpoptApplicationConstructs) {
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    ASSERT_TRUE(Ipopt::IsValid(app));
}

TEST(SolverLink, NloptOptConstructs) {
    nlopt::opt opt(nlopt::LN_COBYLA, 2);
    EXPECT_EQ(opt.get_dimension(), 2u);
}
```

- [ ] **Step 4: Wire pkg-config discovery + the test target in CMake**

Append to `CMakeLists.txt` (after the `goss_nlp_tests` block):

```cmake
# ---- Solver layer: IPOPT + NLopt via pkg-config ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT REQUIRED ipopt)
pkg_check_modules(NLOPT REQUIRED nlopt)

add_executable(goss_solver_tests
  tests/solver/test_solver_link.cpp)
target_include_directories(goss_solver_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/tests ${IPOPT_INCLUDE_DIRS} ${NLOPT_INCLUDE_DIRS})
target_link_libraries(goss_solver_tests PRIVATE
  ${IPOPT_LIBRARIES} ${NLOPT_LIBRARIES} GTest::gtest_main)
target_link_directories(goss_solver_tests PRIVATE
  ${IPOPT_LIBRARY_DIRS} ${NLOPT_LIBRARY_DIRS})
gtest_discover_tests(goss_solver_tests)
```

**Implementer note:** pkg-config for IPOPT 3.11.9 may report include dir as `/usr/include/coin-or` OR `/usr/include`; the `#include <coin-or/IpTNLP.hpp>` form works with `-I/usr/include`. If the link test fails to find the header, inspect `pkg-config --cflags ipopt` inside the container and adjust the include form to match (some builds want `<IpTNLP.hpp>` with `-I/usr/include/coin-or`). Document whichever form works.

- [ ] **Step 5: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R SolverLink --output-on-failure'`
Expected: both SolverLink tests PASS. Confirm no regression: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'` (should be 30 existing + 2 new = 32).

- [ ] **Step 6: Commit**

```bash
git add Dockerfile CMakeLists.txt tests/solver/test_solver_link.cpp
git commit -m "build: add IPOPT + NLopt to container and wire pkg-config discovery"
```

---

### Task 2: Solver interface — SolverStatus, SolverResult, abstract Solver, SolverError

**Files:**
- Create: `include/goss/solver/errors.hpp`
- Create: `include/goss/solver/solver_result.hpp`
- Create: `include/goss/solver/solver.hpp`
- Create: `tests/solver/test_solver_interface.cpp`
- Modify: `CMakeLists.txt` (add the test file + a `goss_solver` lib placeholder)

**Interfaces:**
- Consumes: `goss::nlp::NLPProblem` (const methods), `goss::nlp` types.
- Produces:
  - `enum class goss::solver::SolverStatus { Success, InfeasibleProblem, IterationLimit, NumericalError, Failure };`
  - `struct goss::solver::SolverResult { SolverStatus status; std::vector<double> x; double objective_value; std::vector<double> constraint_multipliers; std::string message; };`
  - `class goss::solver::SolverError : public std::runtime_error` (ctor from `const std::string&`).
  - Abstract `class goss::solver::Solver` with: protected default ctor; deleted copy/move; `virtual ~Solver() = default;` `virtual SolverResult solve(const nlp::NLPProblem& problem, const std::vector<double>& initial_guess) = 0;`

- [ ] **Step 1: Write the failing contract test**

```cpp
// tests/solver/test_solver_interface.cpp
#include <gtest/gtest.h>
#include <memory>
#include "goss/solver/solver.hpp"
#include "goss/solver/solver_result.hpp"
#include "goss/solver/errors.hpp"

namespace {
// Minimal stub proving the interface is implementable + polymorphic.
class StubSolver : public goss::solver::Solver {
 public:
    goss::solver::SolverResult solve(const goss::nlp::NLPProblem&,
                                     const std::vector<double>& initial_guess) override {
        goss::solver::SolverResult result;
        result.status = goss::solver::SolverStatus::Success;
        result.x = initial_guess;
        result.objective_value = 0.0;
        return result;
    }
};
}  // namespace

TEST(SolverInterface, StatusEnumHasExpectedValues) {
    EXPECT_NE(goss::solver::SolverStatus::Success, goss::solver::SolverStatus::Failure);
}

TEST(SolverInterface, SolverErrorIsThrowable) {
    EXPECT_THROW(throw goss::solver::SolverError("boom"), goss::solver::SolverError);
}

TEST(SolverInterface, IsPolymorphic) {
    std::unique_ptr<goss::solver::Solver> solver = std::make_unique<StubSolver>();
    auto result = solver->solve(*static_cast<goss::nlp::NLPProblem*>(nullptr), {1.0, 2.0});
    // Note: StubSolver never dereferences the problem, so the null cast is safe HERE only.
    EXPECT_EQ(result.status, goss::solver::SolverStatus::Success);
    ASSERT_EQ(result.x.size(), 2u);
}
```

**Implementer note:** the null-cast in `IsPolymorphic` is a deliberate shortcut valid ONLY because `StubSolver::solve` ignores the problem argument. If it feels too hacky, instead build a trivial real `NLPProblem` from a `CppADCGBackend` (see Task 4's fixtures) — but for a pure interface test the stub is acceptable. Prefer the real-problem form if the reviewer would object; both are fine.

- [ ] **Step 2: Write the headers**

```cpp
// include/goss/solver/errors.hpp
#pragma once
#include <stdexcept>
#include <string>
namespace goss::solver {
class SolverError : public std::runtime_error {
 public:
    explicit SolverError(const std::string& message) : std::runtime_error(message) {}
};
}  // namespace goss::solver
```

```cpp
// include/goss/solver/solver_result.hpp
#pragma once
#include <string>
#include <vector>
namespace goss::solver {
enum class SolverStatus {
    Success,            // converged to an optimal (or acceptable) point
    InfeasibleProblem,  // solver proved / detected infeasibility
    IterationLimit,     // hit max iterations / evaluations without converging
    NumericalError,     // NaN / invalid number / roundoff-limited
    Failure             // any other unrecoverable failure
};
struct SolverResult {
    SolverStatus status = SolverStatus::Failure;
    std::vector<double> x;                        // final primal solution
    double objective_value = 0.0;                 // objective at x
    std::vector<double> constraint_multipliers;   // λ (may be empty for derivative-free)
    std::string message;                          // human-readable solver status text
};
}  // namespace goss::solver
```

```cpp
// include/goss/solver/solver.hpp
#pragma once
#include <vector>
#include "goss/nlp/nlp_problem.hpp"
#include "goss/solver/solver_result.hpp"
namespace goss::solver {
/// Abstract solver over an NLPProblem. Concrete adapters (IPOPT, NLopt) must
/// confine all third-party solver types to their own .cpp — no solver library
/// type may appear in this or any public header.
class Solver {
 public:
    virtual ~Solver() = default;
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;

    /// Solve the problem from initial_guess. Solve OUTCOMES (including
    /// non-convergence) are reported via the returned SolverResult.status;
    /// only setup/usage errors throw (SolverError).
    virtual SolverResult solve(const nlp::NLPProblem& problem,
                               const std::vector<double>& initial_guess) = 0;

 protected:
    Solver() = default;
};
}  // namespace goss::solver
```

- [ ] **Step 3: Wire CMake**

Append to the solver section of `CMakeLists.txt`:
```cmake
add_library(goss_solver STATIC src/solver/ipopt_solver.cpp src/solver/nlopt_solver.cpp)
target_include_directories(goss_solver PUBLIC ${CMAKE_SOURCE_DIR}/include)
# Task 1 created goss_ipopt_iface / goss_nlopt_iface INTERFACE targets that carry
# the include dirs, -DHAVE_CSTDDEF (IPOPT 3.11.9 headers #error without it), and
# link flags. Link them PRIVATE so goss_solver inherits everything in one line.
target_link_libraries(goss_solver PUBLIC goss_nlp goss_ad
  PRIVATE goss_ipopt_iface goss_nlopt_iface)
```
This references `src/solver/ipopt_solver.cpp` and `src/solver/nlopt_solver.cpp`, which do not exist yet (Tasks 3 and 6). To keep the build green NOW, create minimal placeholder .cpp files each containing only a comment and an `#include "goss/solver/solver.hpp"`. They are filled in later tasks.

Add `tests/solver/test_solver_interface.cpp` to the `goss_solver_tests` sources, and link `goss_solver`:
```cmake
target_link_libraries(goss_solver_tests PRIVATE goss_solver ...existing...)
```

- [ ] **Step 4: Create placeholder .cpp files**

```cpp
// src/solver/ipopt_solver.cpp
#include "goss/solver/solver.hpp"
// IpoptSolver + IpoptTNLPAdapter implemented in Task 3.
```
```cpp
// src/solver/nlopt_solver.cpp
#include "goss/solver/solver.hpp"
// NloptSolver implemented in Task 6.
```

- [ ] **Step 5: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake -S . -B build && cmake --build build && ctest --test-dir build -R SolverInterface --output-on-failure'`
Expected: 3 SolverInterface tests PASS; full suite still green.

- [ ] **Step 6: Commit**

```bash
git add include/goss/solver/ src/solver/ tests/solver/test_solver_interface.cpp CMakeLists.txt
git commit -m "feat: solver interface with SolverStatus, SolverResult, SolverError"
```

---
### Task 3: IpoptSolver + IpoptTNLPAdapter (the TNLP translation)

**Files:**
- Create: `include/goss/solver/ipopt_solver.hpp`
- Modify: `src/solver/ipopt_solver.cpp` (replace placeholder)
- Create: `tests/solver/test_ipopt_solver.cpp`
- Modify: `CMakeLists.txt` (add the test file to goss_solver_tests)

**Interfaces:**
- Consumes: `Solver`, `SolverResult`, `nlp::NLPProblem` (all its eval methods + bounds + num_variables/num_constraints + constraint_jacobian_sparsity + lagrangian_hessian_sparsity). IPOPT `TNLP`/`IpoptApplication` (confined to the .cpp).
- Produces:
  - `class goss::solver::IpoptSolver : public Solver` with a default constructor, an optional `set_option`-style config, and `SolverResult solve(const nlp::NLPProblem&, const std::vector<double>& initial_guess) override;`. Header exposes NO IPOPT type.
  - Internal (in .cpp only) `class IpoptTNLPAdapter : public Ipopt::TNLP` wrapping a `const NLPProblem&` + the initial guess, implementing all 8 TNLP methods + finalize_solution, writing the result into a `SolverResult` the solver reads back.

**TNLP method mapping (verified signatures):**
- `get_nlp_info(n, m, nnz_jac_g, nnz_h_lag, index_style)`: n=num_variables, m=num_constraints, nnz_jac_g=`constraint_jacobian_sparsity().size()`, nnz_h_lag=`lagrangian_hessian_sparsity().size()`, index_style=`C_STYLE`.
- `get_bounds_info(n, x_l, x_u, m, g_l, g_u)`: copy from `variable_lower/upper_bounds()` and `constraint_lower/upper_bounds()`.
- `get_starting_point(...)`: copy `initial_guess` into `x` when `init_x`; return false paths for z/lambda (return true, only fill x).
- `eval_f(n, x, new_x, obj_value)`: `obj_value = problem.eval_objective(vec(x,n))`.
- `eval_grad_f(n, x, new_x, grad_f)`: copy `problem.eval_objective_gradient(vec(x,n))` (dense, size n) into grad_f.
- `eval_g(n, x, new_x, m, g)`: copy `problem.eval_constraints(vec(x,n))` into g.
- `eval_jac_g(...)`: STRUCTURE call (values==NULL) → write iRow[k]/jCol[k] from `constraint_jacobian_sparsity()[k]` (.first=row, .second=col). VALUES call → copy `problem.eval_constraint_jacobian(vec(x,n))` into values (aligned to the same pattern order).
- `eval_h(...)`: STRUCTURE call → write iRow/jCol from `lagrangian_hessian_sparsity()[k]` (already lower-triangle). VALUES call → copy `problem.eval_lagrangian_hessian(vec(x,n), obj_factor, vec(lambda,m))` into values.
- `finalize_solution(status, n, x, z_L, z_U, m, g, lambda, obj_value, ...)`: record x, obj_value, lambda into the stored SolverResult; map `Ipopt::SolverReturn` → `SolverStatus`.

**Status mapping (SolverReturn → SolverStatus):** SUCCESS/STOP_AT_ACCEPTABLE_POINT→Success; MAXITER_EXCEEDED→IterationLimit; LOCAL_INFEASIBILITY→InfeasibleProblem; INVALID_NUMBER_DETECTED→NumericalError; everything else→Failure. Also map the `ApplicationReturnStatus` from `OptimizeTNLP`: if Initialize() or Optimize returns a hard error (e.g. Invalid_Option), throw SolverError; a solve that runs to finalize_solution reports via the recorded status.

- [ ] **Step 1: Write the failing test (a QP with known optimum)**

Use a tiny self-contained problem first, before HS. Minimize x0²+x1² s.t. x0+x1=1 (the Task-2 nlp QuadraticWithLinearConstraint packed fixture, but here as a solver test). Optimum: x*=(0.5,0.5), f*=0.5.

```cpp
// tests/solver/test_ipopt_solver.cpp
#include <gtest/gtest.h>
#include <memory>
#include "goss/solver/ipopt_solver.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/cppadcg_backend.hpp"

namespace {
// objective x0^2+x1^2 (output 0); equality constraint x0+x1-1 (output 1, bounds [0,0]).
struct EqQP {
    std::size_t input_size() const { return 2; }
    std::size_t output_size() const { return 2; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        return {x[0]*x[0] + x[1]*x[1], x[0] + x[1] - T(1)};
    }
};
goss::nlp::NLPProblem make_eq_qp(const std::string& name) {
    EqQP f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), name);
    return goss::nlp::NLPProblem(std::move(backend),
        {-10.0, -10.0}, {10.0, 10.0}, {0.0}, {0.0});
}
}  // namespace

TEST(IpoptSolver, SolvesEqualityConstrainedQP) {
    auto problem = make_eq_qp("ipopt_qp");
    goss::solver::IpoptSolver solver;
    auto result = solver.solve(problem, {2.0, -1.0});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    ASSERT_EQ(result.x.size(), 2u);
    EXPECT_NEAR(result.x[0], 0.5, 1e-5);
    EXPECT_NEAR(result.x[1], 0.5, 1e-5);
    EXPECT_NEAR(result.objective_value, 0.5, 1e-6);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — `ipopt_solver.hpp` / IpoptSolver not defined.

- [ ] **Step 3: Write the header**

```cpp
// include/goss/solver/ipopt_solver.hpp
#pragma once
#include <vector>
#include "goss/solver/solver.hpp"
namespace goss::solver {
/// IPOPT-backed solver. All IPOPT types are confined to the .cpp.
class IpoptSolver : public Solver {
 public:
    IpoptSolver() = default;
    /// Convergence tolerance passed to IPOPT (option "tol"). Default 1e-8.
    void set_tolerance(double tolerance) { tolerance_ = tolerance; }
    /// IPOPT print_level (0 = silent .. 12). Default 0.
    void set_print_level(int level) { print_level_ = level; }
    /// Max iterations (option "max_iter"). Default 3000.
    void set_max_iterations(int iterations) { max_iterations_ = iterations; }

    SolverResult solve(const nlp::NLPProblem& problem,
                       const std::vector<double>& initial_guess) override;

 private:
    double tolerance_ = 1e-8;
    int print_level_ = 0;
    int max_iterations_ = 3000;
};
}  // namespace goss::solver
```

- [ ] **Step 4: Write the adapter + solver in the .cpp**

Replace the placeholder `src/solver/ipopt_solver.cpp`. Structure:
- `#include <IpTNLP.hpp>`, `<IpIpoptApplication.hpp>`, `<IpSolveStatistics.hpp>`, the header, `<vector>`. **NOTE (verified in Task 1):** IPOPT 3.11.9 headers are under `/usr/include/coin/` (NOT `coin-or/`), so the include form has NO prefix — `<IpTNLP.hpp>`, not `<coin-or/IpTNLP.hpp>`. The `goss_ipopt_iface` target (linked by `goss_solver`) supplies the `-I/usr/include/coin` include dir and `-DHAVE_CSTDDEF`.
- Anonymous-namespace helper `std::vector<double> to_vector(const Ipopt::Number* p, Ipopt::Index n)`.
- `class IpoptTNLPAdapter : public Ipopt::TNLP` holding `const nlp::NLPProblem& problem_`, `const std::vector<double>& initial_guess_`, and `SolverResult& result_` (reference to the solver's result). Implement the 9 methods per the mapping above. In the STRUCTURE branch of eval_jac_g/eval_h, iterate the corresponding sparsity() pattern; in the VALUES branch, call the eval_* method and memcpy/loop into `values`.
- The status map from `Ipopt::SolverReturn` in finalize_solution.
- `IpoptSolver::solve`: construct `SolverResult result;` set default status Failure; `Ipopt::SmartPtr<Ipopt::TNLP> adapter = new IpoptTNLPAdapter(problem, initial_guess, result);` create app via `IpoptApplicationFactory()`; set options (tol, print_level, max_iter, `hessian_approximation=exact`, `mu_strategy=adaptive`); `if (app->Initialize() != Ipopt::Solve_Succeeded) throw SolverError(...)`; `app->OptimizeTNLP(adapter);` then return `result` (finalize_solution already populated it). If OptimizeTNLP returns a config-level hard failure (Invalid_Option/Invalid_Problem_Definition/Unrecoverable_Exception) AND finalize_solution was never called, set result.status=Failure with the message.

**Implementer note (verified):** IPOPT 3.11.9 does NOT have `Maximum_WallTime_Exceeded` — do not reference it. Only lower-triangular Hessian entries are expected — `lagrangian_hessian_sparsity()` already returns lower-triangle, so pass through directly. `SmartPtr` owns the adapter — heap-allocate with `new`, never delete it. Use `IsValid()` to check SmartPtrs. The result reference captured by the adapter must outlive the solve (it is a local in `solve()`, and the adapter does not outlive the solve, so this is safe).

- [ ] **Step 5: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "IpoptSolver.SolvesEqualityConstrainedQP" --output-on-failure'`
Expected: PASS — x*≈(0.5,0.5), f*≈0.5. If IPOPT prints noise, ensure print_level defaulted to 0.

- [ ] **Step 6: Commit**

```bash
git add include/goss/solver/ipopt_solver.hpp src/solver/ipopt_solver.cpp tests/solver/test_ipopt_solver.cpp CMakeLists.txt
git commit -m "feat: IpoptSolver via TNLP adapter, solving a constrained QP"
```

---

### Task 4: Hock-Schittkowski fixtures + IPOPT solves the classic HS71

**Files:**
- Create: `tests/solver/hs_fixtures.hpp`
- Modify: `tests/solver/test_ipopt_solver.cpp`

**Interfaces:**
- Consumes: `NLPProblem`, `CppADCGBackend`, `IpoptSolver`.
- Produces: packed HS functors (objective in output 0, constraints in 1..m) + per-problem `NLPProblem` builders with the correct bounds and constraint bounds encoding the HS constraint conventions. Start with HS71 (the canonical IPOPT example) plus HS28 (equality QP) and HS35 (inequality QP).

**Verified HS specs (oracles):**
- **HS71**: min x1·x4·(x1+x2+x3)+x3; ineq x1·x2·x3·x4 ≥ 25; eq x1²+x2²+x3²+x4²=40; 1≤xi≤5; x0=(1,5,5,1); x*≈(1.0,4.743,3.821,1.379); f*≈17.0140173.
- **HS28**: min (x1+x2)²+(x2+x3)²; eq x1+2x2+3x3=1; no bounds; x0=(-4,1,1); x*=(0.5,-0.5,0.5); f*=0.
- **HS35**: min 9−8x1−6x2−4x3+2x1²+2x2²+x3²+2x1x2+2x1x3; ineq x1+x2+2x3≤3; xi≥0; x0=(0.5,0.5,0.5); f*=1/9≈0.1111.

**Encoding into NLPProblem (packing + bounds):** Each functor packs the objective in output 0 and constraints in outputs 1..m. Constraint bounds encode the sense:
- Equality h(x)=0 → pack `h(x)` as an output, constraint bound [0, 0].
- Inequality g(x) ≥ c → pack `g(x)` as an output, constraint bound [c, +INF] (use 2e19 for +INF).
- Inequality g(x) ≤ c → pack `g(x)`, constraint bound [-INF, c].
Variable bounds go in the variable bound vectors (use ±2e19 for free variables, e.g. HS28).

- [ ] **Step 1: Write the HS71 fixture + failing solve test**

```cpp
// tests/solver/hs_fixtures.hpp
#pragma once
#include <cstddef>
#include <vector>
namespace goss::solver::hs {
constexpr double kInf = 2e19;
// HS71: outputs = [objective, ineq g=x1x2x3x4, eq h=sum xi^2].
struct HS71 {
    std::size_t input_size() const { return 4; }
    std::size_t output_size() const { return 3; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        T obj = x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2];
        T g   = x[0]*x[1]*x[2]*x[3];
        T h   = x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3];
        return {obj, g, h};
    }
};
}  // namespace goss::solver::hs
```

```cpp
// append to tests/solver/test_ipopt_solver.cpp
#include "solver/hs_fixtures.hpp"

TEST(IpoptSolver, SolvesHS71) {
    goss::solver::hs::HS71 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "ipopt_hs71");
    // vars 1..5 ; ineq g >= 25 -> [25, INF]; eq h == 40 -> [40, 40].
    goss::nlp::NLPProblem problem(std::move(backend),
        {1.0, 1.0, 1.0, 1.0}, {5.0, 5.0, 5.0, 5.0},
        {25.0, 40.0}, {goss::solver::hs::kInf, 40.0});
    goss::solver::IpoptSolver solver;
    auto result = solver.solve(problem, {1.0, 5.0, 5.0, 1.0});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 17.0140173, 1e-4);
    EXPECT_NEAR(result.x[0], 1.0, 1e-3);
    EXPECT_NEAR(result.x[1], 4.743, 1e-3);
    EXPECT_NEAR(result.x[2], 3.821, 1e-3);
    EXPECT_NEAR(result.x[3], 1.379, 1e-3);
}
```

- [ ] **Step 2: Run to verify it fails, then passes**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "IpoptSolver.SolvesHS71" --output-on-failure'`
Expected: PASS (the adapter from Task 3 already handles this; the fixture is the new part). If it fails, check the constraint-bound encoding (g≥25 is [25, INF]; h=40 is [40,40]) and that kInf (2e19) is used for the one-sided bound.

- [ ] **Step 3: Add HS28 (equality) and HS35 (inequality) fixtures + tests**

```cpp
// append to tests/solver/hs_fixtures.hpp (inside namespace)
// HS28: outputs = [objective, eq h=x1+2x2+3x3-1]. f*=0 at (0.5,-0.5,0.5).
struct HS28 {
    std::size_t input_size() const { return 3; }
    std::size_t output_size() const { return 2; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        T obj = (x[0]+x[1])*(x[0]+x[1]) + (x[1]+x[2])*(x[1]+x[2]);
        T h   = x[0] + T(2)*x[1] + T(3)*x[2] - T(1);
        return {obj, h};
    }
};
// HS35: outputs = [objective, ineq g=x1+x2+2x3 (<=3)]. f*=1/9. xi>=0.
struct HS35 {
    std::size_t input_size() const { return 3; }
    std::size_t output_size() const { return 2; }
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        T obj = T(9) - T(8)*x[0] - T(6)*x[1] - T(4)*x[2]
              + T(2)*x[0]*x[0] + T(2)*x[1]*x[1] + x[2]*x[2]
              + T(2)*x[0]*x[1] + T(2)*x[0]*x[2];
        T g   = x[0] + x[1] + T(2)*x[2];
        return {obj, g};
    }
};
```

```cpp
// append to tests/solver/test_ipopt_solver.cpp
TEST(IpoptSolver, SolvesHS28) {
    goss::solver::hs::HS28 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "ipopt_hs28");
    // no var bounds (free) -> ±INF; eq h==0 -> [0,0].
    const double inf = goss::solver::hs::kInf;
    goss::nlp::NLPProblem problem(std::move(backend),
        {-inf, -inf, -inf}, {inf, inf, inf}, {0.0}, {0.0});
    goss::solver::IpoptSolver solver;
    auto result = solver.solve(problem, {-4.0, 1.0, 1.0});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 0.0, 1e-6);
}

TEST(IpoptSolver, SolvesHS35) {
    goss::solver::hs::HS35 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "ipopt_hs35");
    // xi>=0 -> [0,INF]; ineq g<=3 -> [-INF,3].
    const double inf = goss::solver::hs::kInf;
    goss::nlp::NLPProblem problem(std::move(backend),
        {0.0, 0.0, 0.0}, {inf, inf, inf}, {-inf}, {3.0});
    goss::solver::IpoptSolver solver;
    auto result = solver.solve(problem, {0.5, 0.5, 0.5});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 1.0 / 9.0, 1e-5);
}
```

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "IpoptSolver" --output-on-failure'`
Expected: all IpoptSolver tests PASS (QP + HS71 + HS28 + HS35).

- [ ] **Step 5: Commit**

```bash
git add tests/solver/hs_fixtures.hpp tests/solver/test_ipopt_solver.cpp
git commit -m "test: IPOPT solves HS71, HS28, HS35 against known optima"
```

---
### Task 5: NloptSolver (COBYLA) — objective + bounds, unconstrained/bound-only first

**Files:**
- Create: `include/goss/solver/nlopt_solver.hpp`
- Modify: `src/solver/nlopt_solver.cpp` (replace placeholder)
- Create: `tests/solver/test_nlopt_solver.cpp`
- Modify: `CMakeLists.txt` (add the test file)

**Interfaces:**
- Consumes: `Solver`, `SolverResult`, `nlp::NLPProblem`. NLopt `nlopt::opt` (confined to the .cpp).
- Produces:
  - `class goss::solver::NloptSolver : public Solver` — default ctor, `set_max_evaluations(int)` (default 20000), `set_xtol_rel(double)` (default 1e-8), and `solve(...) override`. Header exposes NO NLopt type.
  - This task handles the OBJECTIVE + variable bounds + the result/status mapping, tested on a bound-only problem (HS1 / Rosenbrock with a bound, m=0). Constraints are added in Task 6.

**NLopt facts (verified):** C++ header `<nlopt.hpp>`. Algorithm `nlopt::LN_COBYLA` (derivative-free, handles nonlinear ineq+eq). Objective callback `double f(const std::vector<double>& x, std::vector<double>& grad, void* data)` — for COBYLA `grad` is empty, do not write it. `set_min_objective`, `set_lower_bounds(vector)`, `set_upper_bounds(vector)`, `set_xtol_rel`, `set_maxeval`. `optimize(vector<double>& x, double& minf)` returns `nlopt::result` (positive=success: SUCCESS=1, STOPVAL=2, FTOL=3, XTOL=4, MAXEVAL=5, MAXTIME=6; negative=failure). NLopt THROWS on failure by default — MUST `set_exceptions_enabled(false)` and inspect the code, or catch. The `f_data` pointer is stored WITHOUT copying — the pointed-to object must outlive `optimize()`.

**Result/status mapping (nlopt::result → SolverStatus):** SUCCESS/STOPVAL_REACHED/FTOL_REACHED/XTOL_REACHED→Success; MAXEVAL_REACHED/MAXTIME_REACHED→IterationLimit; FAILURE/INVALID_ARGS→Failure; OUT_OF_MEMORY→Failure; ROUNDOFF_LIMITED→NumericalError; FORCED_STOP→Failure. Catch `std::exception` from NLopt and set status=NumericalError (roundoff) or Failure with the message; do not let it escape.

**±INF bounds:** NLopt accepts `HUGE_VAL`/`std::numeric_limits<double>::infinity()` for unbounded. The NLPProblem stores ±2e19 for "free"; convert bounds >= 1e19 to +infinity and <= -1e19 to -infinity before passing to NLopt (COBYLA behaves better with true infinities than with 2e19).

- [ ] **Step 1: Write the failing test (bound-only Rosenbrock, HS1)**

```cpp
// tests/solver/test_nlopt_solver.cpp
#include <gtest/gtest.h>
#include <memory>
#include "goss/solver/nlopt_solver.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/cppadcg_backend.hpp"

namespace {
// HS1: min 100(x2-x1^2)^2 + (1-x1)^2 ; x2 >= -1.5 ; x* = (1,1), f* = 0. m=0.
struct HS1 {
    std::size_t input_size() const { return 2; }
    std::size_t output_size() const { return 1; }  // objective only, no constraints
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x) const {
        T a = x[1] - x[0]*x[0];
        T b = T(1) - x[0];
        return {T(100)*a*a + b*b};
    }
};
}  // namespace

TEST(NloptSolver, SolvesBoundConstrainedRosenbrockHS1) {
    HS1 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "nlopt_hs1");
    const double inf = 2e19;
    // x1 free, x2 >= -1.5 ; no general constraints (m=0).
    goss::nlp::NLPProblem problem(std::move(backend),
        {-inf, -1.5}, {inf, inf}, {}, {});
    goss::solver::NloptSolver solver;
    solver.set_max_evaluations(50000);
    auto result = solver.solve(problem, {-1.2, 1.0});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 0.0, 1e-4);
    EXPECT_NEAR(result.x[0], 1.0, 1e-2);
    EXPECT_NEAR(result.x[1], 1.0, 1e-2);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build 2>&1 | tail -20'`
Expected: FAIL — nlopt_solver.hpp / NloptSolver not defined.

- [ ] **Step 3: Write the header**

```cpp
// include/goss/solver/nlopt_solver.hpp
#pragma once
#include <vector>
#include "goss/solver/solver.hpp"
namespace goss::solver {
/// NLopt COBYLA (derivative-free) solver. All NLopt types confined to the .cpp.
/// Serves as an independent baseline to cross-check gradient/Hessian-based
/// solvers (IPOPT) without relying on the AD derivative path.
class NloptSolver : public Solver {
 public:
    NloptSolver() = default;
    void set_max_evaluations(int evaluations) { max_evaluations_ = evaluations; }
    void set_xtol_rel(double tolerance) { xtol_rel_ = tolerance; }

    SolverResult solve(const nlp::NLPProblem& problem,
                       const std::vector<double>& initial_guess) override;

 private:
    int max_evaluations_ = 20000;
    double xtol_rel_ = 1e-8;
};
}  // namespace goss::solver
```

- [ ] **Step 4: Implement (.cpp) — objective + bounds + status only**

Replace the placeholder `src/solver/nlopt_solver.cpp`. Structure:
- `#include <nlopt.hpp>`, `<cmath>`, `<limits>`, the header.
- A helper `std::vector<double> clamp_infinities(const std::vector<double>& bounds, bool lower)` converting ±2e19 to ±infinity.
- A data struct passed as `void*` to the callback: `struct ObjectiveData { const nlp::NLPProblem* problem; };` (COBYLA calls the objective; the callback computes `problem->eval_objective(x)`; leave grad untouched since it's empty for COBYLA).
- `NloptSolver::solve`:
  - Validate `initial_guess.size() == problem.num_variables()` else throw SolverError.
  - `nlopt::opt optimizer(nlopt::LN_COBYLA, problem.num_variables());`
  - `optimizer.set_exceptions_enabled(false);` (CRITICAL — inspect return codes, don't let NLopt throw).
  - Set lower/upper bounds via clamp_infinities.
  - `ObjectiveData data{&problem};` then `optimizer.set_min_objective(&objective_callback, &data);` (data is a local — it outlives optimize(), which is called next in the same scope; safe).
  - `optimizer.set_xtol_rel(xtol_rel_); optimizer.set_maxeval(max_evaluations_);`
  - `std::vector<double> x = initial_guess; double minf = 0.0;`
  - Wrap `nlopt::result code = optimizer.optimize(x, minf);` in try/catch (belt-and-suspenders even with exceptions disabled); on catch set NumericalError/Failure.
  - Build SolverResult: x, objective_value=minf, status from the mapping, message from a code→string helper. constraint_multipliers stays empty (COBYLA doesn't produce them).
- Free function objective_callback with the exact vfunc signature; casts `void* data` back to `ObjectiveData*` and returns `d->problem->eval_objective(x)`.

**Implementer note:** with `set_exceptions_enabled(false)`, `optimize()` returns a negative `nlopt::result` instead of throwing; still keep a try/catch around it for robustness. Constraints are NOT added in this task — this test problem has m=0. If the NLPProblem has constraints and this task's code ignores them, that's fine ONLY because the test uses m=0; Task 6 adds constraint handling.

- [ ] **Step 5: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "NloptSolver.SolvesBoundConstrainedRosenbrockHS1" --output-on-failure'`
Expected: PASS (COBYLA converges to (1,1); allow loose 1e-2 tolerance).

- [ ] **Step 6: Commit**

```bash
git add include/goss/solver/nlopt_solver.hpp src/solver/nlopt_solver.cpp tests/solver/test_nlopt_solver.cpp CMakeLists.txt
git commit -m "feat: NloptSolver (COBYLA) for objective + bounds, solving HS1"
```

---

### Task 6: NloptSolver constraint handling (two-sided decomposition)

**Files:**
- Modify: `src/solver/nlopt_solver.cpp`
- Modify: `tests/solver/test_nlopt_solver.cpp`

**Interfaces:**
- Consumes: `NLPProblem::constraint_lower_bounds()/upper_bounds()`, `eval_constraints`. NLopt `add_inequality_mconstraint`.
- Produces: NloptSolver that handles general two-sided constraints `gL <= g(x) <= gU` by decomposition, tested on HS71 and HS35.

**Decomposition (verified NLopt convention `fc(x) <= 0`):** For each constraint i with bounds [gL_i, gU_i]:
- If gL_i == gU_i (equality): add an equality m-constraint `g_i(x) - gL_i == 0`.
- Else: for each finite side, add an inequality: upper `g_i(x) - gU_i <= 0` (when gU_i < INF), lower `gL_i - g_i(x) <= 0` (when gL_i > -INF).
Simplest robust approach: build ONE inequality m-constraint that stacks all finite one-sided rows, and ONE equality m-constraint for the equality rows. COBYLA supports both `add_inequality_mconstraint` and `add_equality_mconstraint`.

- [ ] **Step 1: Write the failing tests (HS71 + HS35 via NLopt)**

```cpp
// append to tests/solver/test_nlopt_solver.cpp
#include "solver/hs_fixtures.hpp"

TEST(NloptSolver, SolvesHS71) {
    goss::solver::hs::HS71 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "nlopt_hs71");
    goss::nlp::NLPProblem problem(std::move(backend),
        {1.0, 1.0, 1.0, 1.0}, {5.0, 5.0, 5.0, 5.0},
        {25.0, 40.0}, {goss::solver::hs::kInf, 40.0});
    goss::solver::NloptSolver solver;
    solver.set_max_evaluations(100000);
    auto result = solver.solve(problem, {1.0, 5.0, 5.0, 1.0});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 17.0140173, 1e-2);  // COBYLA: looser tol
}

TEST(NloptSolver, SolvesHS35) {
    goss::solver::hs::HS35 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), "nlopt_hs35");
    const double inf = goss::solver::hs::kInf;
    goss::nlp::NLPProblem problem(std::move(backend),
        {0.0, 0.0, 0.0}, {inf, inf, inf}, {-inf}, {3.0});
    goss::solver::NloptSolver solver;
    solver.set_max_evaluations(100000);
    auto result = solver.solve(problem, {0.5, 0.5, 0.5});
    ASSERT_EQ(result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(result.objective_value, 1.0 / 9.0, 1e-3);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "NloptSolver.SolvesHS71" --output-on-failure'`
Expected: FAIL — HS71 has constraints NLopt currently ignores, so it converges to the wrong point / wrong status.

- [ ] **Step 3: Add constraint decomposition to the .cpp**

Extend `NloptSolver::solve` after setting the objective. Build two index lists from `problem.constraint_lower_bounds()/upper_bounds()`:
- Equality rows (gL==gU) and their target values.
- One-sided inequality rows: for each constraint, up to two entries (upper: (i, +1, gU); lower: (i, -1, gL)) for each FINITE side (treat |bound| >= 1e19 as infinite → skip that side).
Store these in a `ConstraintData` struct (holding `const NLPProblem* problem` + the row/sign/target lists) passed as `void*`. Implement two free m-constraint callbacks:
- `inequality_mconstraint(unsigned m, double* result, unsigned n, const double* x, double* grad, void* data)`: compute `g = problem->eval_constraints(x_vec)`; for each stacked inequality entry k=(row, sign, target): `result[k] = sign * (g[row] - target)` if sign==+1 means `g[row]-gU <=0`; for the lower side we want `gL - g[row] <= 0` so store sign=-1 and compute `result[k] = target - g[row]` — define consistently. Leave grad null (COBYLA).
- `equality_mconstraint(...)`: `result[j] = g[eq_row_j] - eq_target_j`.
Register with `optimizer.add_inequality_mconstraint(&inequality_mconstraint, &cdata, tol_vec)` and `add_equality_mconstraint(...)` with per-row tol (e.g. 1e-8), only if the respective count > 0. The ConstraintData local must outlive optimize() (same scope — safe).

**Implementer note:** define the inequality result sign carefully and consistently: NLopt wants `result[k] <= 0` feasible. For an upper bound gU: feasible when `g <= gU` ⇒ `result = g - gU`. For a lower bound gL: feasible when `g >= gL` ⇒ `result = gL - g`. Get this right or constraints invert and the solver finds a wrong optimum. The HS71 test (with its known optimum) is the check.

- [ ] **Step 4: Build and run — verify pass**

Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "NloptSolver" --output-on-failure'`
Expected: all NloptSolver tests PASS (HS1 + HS71 + HS35). HS71 to 1e-2, HS35 to 1e-3.

- [ ] **Step 5: Commit**

```bash
git add src/solver/nlopt_solver.cpp tests/solver/test_nlopt_solver.cpp
git commit -m "feat: NloptSolver constraint decomposition, solving HS71 and HS35"
```

---

### Task 7: Cross-solver agreement test

**Files:**
- Create: `tests/solver/test_solver_agreement.cpp`
- Modify: `CMakeLists.txt` (add the test file)

**Interfaces:**
- Consumes: `IpoptSolver`, `NloptSolver`, the HS fixtures, `NLPProblem`.
- Produces: no production code — a test proving both solvers reach the same optimum on the same problem, which is the whole point of having a derivative-free baseline (it validates the model + IPOPT's gradient/Hessian path independently).

- [ ] **Step 1: Write the agreement test**

```cpp
// tests/solver/test_solver_agreement.cpp
#include <gtest/gtest.h>
#include <memory>
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/nlopt_solver.hpp"
#include "goss/nlp/nlp_problem.hpp"
#include "goss/ad/cppadcg_backend.hpp"
#include "solver/hs_fixtures.hpp"

namespace {
goss::nlp::NLPProblem make_hs71(const std::string& name) {
    goss::solver::hs::HS71 f;
    auto backend = std::make_unique<goss::ad::CppADCGBackend>(f, f.input_size(), name);
    return goss::nlp::NLPProblem(std::move(backend),
        {1.0, 1.0, 1.0, 1.0}, {5.0, 5.0, 5.0, 5.0},
        {25.0, 40.0}, {goss::solver::hs::kInf, 40.0});
}
}  // namespace

TEST(SolverAgreement, IpoptAndNloptReachSameOptimumOnHS71) {
    auto problem_ipopt = make_hs71("agree_hs71_ipopt");
    auto problem_nlopt = make_hs71("agree_hs71_nlopt");
    const std::vector<double> x0{1.0, 5.0, 5.0, 1.0};

    goss::solver::IpoptSolver ipopt;
    auto r_ipopt = ipopt.solve(problem_ipopt, x0);

    goss::solver::NloptSolver nlopt;
    nlopt.set_max_evaluations(100000);
    auto r_nlopt = nlopt.solve(problem_nlopt, x0);

    ASSERT_EQ(r_ipopt.status, goss::solver::SolverStatus::Success);
    ASSERT_EQ(r_nlopt.status, goss::solver::SolverStatus::Success);
    // Independent solvers, independent derivative paths → same objective.
    EXPECT_NEAR(r_ipopt.objective_value, r_nlopt.objective_value, 1e-2);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(r_ipopt.x[i], r_nlopt.x[i], 5e-2) << "x[" << i << "] disagrees";
    }
}
```

- [ ] **Step 2: Wire the test into CMake + build and run**

Add `tests/solver/test_solver_agreement.cpp` to `goss_solver_tests` sources.
Run: `scripts/dev.sh 'cmake --build build && ctest --test-dir build -R "SolverAgreement" --output-on-failure'`
Expected: PASS — both reach f*≈17.014 and the same x*.

- [ ] **Step 3: Run the FULL suite**

Run: `scripts/dev.sh 'ctest --test-dir build --output-on-failure'`
Expected: ALL tests pass (30 prior + solver: link 2 + interface 3 + ipopt 4 + nlopt 3 + agreement 1 = ~43).

- [ ] **Step 4: Commit**

```bash
git add tests/solver/test_solver_agreement.cpp CMakeLists.txt
git commit -m "test: IPOPT and NLopt agree on HS71 optimum (cross-solver check)"
```

---

## Self-Review

**Spec coverage (solver/ layer portion of the design spec):**
- "Solver interface + IpoptAdapter end-to-end" → Tasks 2 (interface), 3 (IPOPT adapter). ✓
- "IPOPT + a derivative-free baseline (NLopt)" → Task 3 (IPOPT), Tasks 5-6 (NLopt/COBYLA). ✓
- "Known-optimum problems: hand-ported Hock-Schittkowski subset" → Task 4 (HS71/28/35 via IPOPT), Task 6 (HS71/35 via NLopt), Task 5 (HS1). ✓
- "IPOPT vs NLopt agreement cross-check" → Task 7. ✓
- "NLPProblem→TNLP mapping validated with real sparse Jac/Hessian" → Task 3 uses exact Hessian via eval_lagrangian_hessian; HS problems exercise real sparse derivatives. ✓
- "derivative-free baseline validates the model independent of exact derivatives" → Tasks 5-7: COBYLA uses only eval_objective/eval_constraints, no gradients. ✓
- Solver-neutral interface (no IPOPT/NLopt in public headers) → Global constraint + Tasks 2/3/5 headers. ✓

**Placeholder scan:** Task 2 creates minimal placeholder .cpp files for ipopt_solver/nlopt_solver so goss_solver links before Tasks 3/6 fill them — each filled in a named later task, standard sequencing. No "TBD"/"add error handling"/"similar to Task N". Every code step has literal content. The one deliberate shortcut (null-cast in SolverInterface.IsPolymorphic) is flagged with an implementer note offering the real-problem alternative.

**Type consistency:**
- `SolverStatus` enum values (Success/InfeasibleProblem/IterationLimit/NumericalError/Failure) defined in Task 2, used in the status maps of Tasks 3, 5, 6.
- `SolverResult` fields (status, x, objective_value, constraint_multipliers, message) consistent across all adapters and tests.
- `Solver::solve(const nlp::NLPProblem&, const std::vector<double>&)` signature identical in the interface (Task 2), both adapter headers (Tasks 3, 5), and every test call.
- HS fixture functors (HS71/HS28/HS35 in hs_fixtures.hpp, HS1 local to nlopt test) use the packing convention (objective output 0, constraints 1..m) consistent with the NLPProblem contract and the constraint-bound encodings in the tests.
- `kInf = 2e19` defined once in hs_fixtures.hpp, used consistently; NLopt adapter converts ±2e19 → ±infinity.
- IPOPT `eval_h` ← `eval_lagrangian_hessian(x, obj_factor, lambda)` and `eval_jac_g` ← `eval_constraint_jacobian` / `constraint_jacobian_sparsity` — names match the merged NLPProblem API exactly (verified against include/goss/nlp/nlp_problem.hpp on main).

**Known external-API notes:** IPOPT 3.11.9 predates `Maximum_WallTime_Exceeded` (flagged, avoided). pkg-config include-dir form for IPOPT may be `/usr/include` vs `/usr/include/coin-or` (flagged in Task 1 with a resolution step). NLopt exceptions disabled via `set_exceptions_enabled(false)` + try/catch (flagged). HS76 excluded (uncertain oracle); HS35 correctly labeled a QP not "Beale".
