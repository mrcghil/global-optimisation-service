# Nonlinear Optimization Framework via Collocation — Design

**Date:** 2026-07-31
**Status:** Approved design, pre-implementation

## 1. Purpose

A C++ framework for solving large nonlinear optimization problems, initially
targeting **optimal control / trajectory optimization** via direct collocation.
The framework must be flexible: a user declares problem structure (states,
controls, dynamics, constraints, cost) and the system transcribes it into a
large sparse Nonlinear Program (NLP), computes fast sparse derivatives, and
hands the result to a pluggable solver (IPOPT, NLopt, ...).

The motivating example: model a queue — add a state (number of people in
queue), add a constraint (state must be non-negative), define dynamics, and
solve.

### Core design goal: minimal rewrite as scope grows

The framework is layered so that the future problem classes below are **additive
features, not rewrites**:

- General sparse NLP (already the core)
- Multi-phase optimal control
- DAE / algebraic constraints
- Hybrid systems / events
- Additional solvers and AD backends

## 2. Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | **C++** | IPOPT is C++ (native linking, no FFI); mature sparse-AD ecosystems; native sparse linear algebra |
| AD backend | **CppAD + CppADCodeGen** | Record tape → detect sparsity → codegen C → JIT compile → native-speed sparse Jacobian/Hessian. Matches the "output a function + jac + hessian" goal exactly |
| AD abstraction | Backend-agnostic interface | Allows benchmarking an Enzyme backend later without touching upper layers |
| Problem class (v1) | Optimal control / trajectory | Queue example fits; other classes layer on top |
| Collocation | **Interface + Trapezoidal + Hermite-Simpson** | Two local schemes prove the transcription abstraction; both highly sparse |
| Solvers (v1) | **IPOPT + a derivative-free baseline (NLopt-style)** | NLopt validates the model independent of the AD path — isolates model bugs from derivative bugs |
| Test framework | **GoogleTest** + FD/complex-step helpers | Widely used, strong CI integration |
| External test suites | **Hand-port a curated ~10-problem Hock-Schittkowski subset** | No parser needed; validates the real DSL code path |

## 3. Architecture

Five layers. **The load-bearing invariant:** everything at or below
`transcription/` speaks only in terms of a flat decision vector `z`, constraint
vector `g(z)`, and their sparse derivatives — never "state", "control", or
"time". This seam is what makes future problem classes additive.

```
Layer 4: Modeling DSL      model/         States, controls, dynamics f(x,u,t),
                                          path/boundary constraints, cost → Problem
Layer 3: Transcription     transcription/ Compiles Problem → NLPProblem
                                          (discretizes time, writes defect constraints).
                                          Problem classes plug in HERE.
Layer 2: NLP core          nlp/           min f(z) s.t. gL≤g(z)≤gU, zL≤z≤zU.
                                          Backend-agnostic. The STABLE core.
Layer 1: AD engine         ad/            CppADCodeGen: record → sparsity detect →
                                          codegen → JIT → sparse ∇f, Jᵍ, ∇²L
Layer 0: Solver adapter    solver/        IPOPT / NLopt / ...
Support:                   sim/           Initial guess & scaling, validation-by-
                                          integration, results/plotting, benchmark
                                          harness, diagnostics
```

### Module responsibilities

| Module | Responsibility | Depends on |
|---|---|---|
| `ad/` | AD backend interface + CppADCodeGen impl: sparse ∇f, Jᵍ, ∇²L | — |
| `nlp/` | `NLPProblem`: flat `z`, variable bounds, `g(z)` bounds, sparse-derivative callbacks | `ad/` |
| `transcription/` | `Transcription` interface + `Trapezoidal`, `HermiteSimpson`; compiles `Problem` → `NLPProblem` | `nlp/` |
| `model/` | DSL: `State`, `Control`, `Dynamics`, `PathConstraint`, `BoundaryConstraint`, `Cost`, `Component` (sub-model composition) → `Problem` | — |
| `solver/` | `Solver` interface + `IpoptAdapter`, `NloptAdapter` | `nlp/` |
| `sim/` | Initial-guess/scaling, validation-by-integration, results/plotting, benchmark harness, diagnostics | all |

### Key interface contracts (the swappable seams)

- **AD interface** (`ad/` ↔ `nlp/`): `eval_f`, `eval_g`, `jac_sparsity`,
  `eval_jac`, `hess_sparsity`, `eval_hess`. CppADCodeGen implements now; Enzyme
  could later.
- **NLPProblem interface** (`nlp/` ↔ `solver/`): exactly the vectors/callbacks
  IPOPT's `TNLP` expects, so solver adapters stay thin.
- **Transcription interface** (`transcription/` ↔ `model/`): `compile(Problem)
  → NLPProblem`. New schemes and new problem classes implement/extend this.

### How future classes attach without rewrite

| Future need | Plugs in at | Core rewrite? |
|---|---|---|
| General NLP | Already `nlp/` — expose directly | None |
| Multi-phase | `transcription/`: multiple phases + linkage constraints | None |
| DAE / algebraic | `transcription/`: algebraic residuals as extra constraints at nodes | None |
| Hybrid / events | `transcription/`: multi-phase + switching constraints | None |
| New solver | `solver/`: new adapter | None |
| New AD backend | `ad/`: new backend behind the interface | None |

## 4. Data Flow (one solve)

```
Problem (model/) --build()--> Transcription.compile (transcription/)
  → NLPProblem (nlp/) records f,g into CppADCodeGen (ad/)
     → sparsity detect → codegen → JIT
  → sim/ builds initial guess z₀ + scaling
  → Solver.solve(NLPProblem, z₀) → z*  (solver calls fast sparse jac/hess)
  → sim/ unpack z* → trajectories, validate-by-integration, plot, diagnostics
```

## 5. Error Handling & Diagnostics

Failures are classified by *where* they occur (following the org's coding
standards: specific exception types, meaningful messages, no silent catch-alls):

- **Setup errors** (at `compile`): dimension mismatches, missing dynamics for a
  declared state, inconsistent bounds → structured exceptions naming the
  offending symbol.
- **AD errors**: sparsity-pattern mismatch or NaN in a derivative → surfaced
  before the solve, never swallowed.
- **Convergence failures** (from solver): diagnostics module inspects solver
  status + KKT residuals and reports likely causes — infeasibility, unbounded
  objective, poor scaling, bad initial guess — rather than a raw status code.

## 6. Testing Strategy

Guiding principle: **each layer has an oracle** (a source of known-correct
answers) so a failure localizes to one layer. Validation is bottom-up — each
layer is validated before the layer above depends on it.

### `ad/` — tested hardest (whole framework trusts these derivatives)
- **Finite differences**: every ∇f, Jacobian, Hessian vs. central differences
  on random inputs (~1e-6). Catches gross errors.
- **Complex-step differentiation**: near-machine-precision gradient oracle with
  no subtractive cancellation — a tighter second opinion than FD.
- **Closed-form cases**: quadratics, Rosenbrock, trig — exact known derivatives.
- **Sparsity correctness**: detected sparsity pattern must equal the known
  pattern for structured functions (banded function ⇒ banded Jacobian). A wrong
  pattern is a silent, dangerous bug.

### `nlp/` — stable core
- Structural/property tests: bounds ordering, dimension consistency, `g(z)`
  bounds match constraint count.
- Round-trip: hand-built tiny NLP returns exactly what `ad/` produced.

### `transcription/` — math of discretization
- **Defect correctness**: feed a known exact trajectory of a simple ODE
  (ẋ=x, ẋ=−x, harmonic oscillator); defects ≈ 0 at the analytic solution.
- **Convergence order**: refine mesh, confirm error shrinks at the theoretical
  rate (trapezoidal O(h²), Hermite-Simpson O(h⁴)). Definitive scheme-correctness
  test.
- **Cross-scheme agreement**: both schemes converge to the same solution.

### `model/` — DSL + composition
- Assembly tests: declarations produce expected NLP dimensions/variable ordering.
- **Composition tests**: name resolution (unresolved/duplicate names error);
  topological ordering of inline expressions; cycle among inline expressions is
  a setup error; inline vs. algebraic flavor of a derived quantity produce
  equivalent optima (inline substitution ≡ algebraic-variable + defining
  constraint) on a small model.
- **Isolated component tests**: a `Component` (e.g. the service-rate model) is
  unit-tested standalone before composition.
- **Queue example as a permanent integration fixture**, built from composed
  components across files.

### `solver/` — adapters + full chain
- **Hock-Schittkowski subset** (hand-ported, ~10 problems): assert we reach
  documented optima + multipliers. Validates model→transcription→nlp→ad→solver.
- **IPOPT vs. NLopt agreement**: both land on the same optimum for well-posed
  problems.

### `sim/` — support stack
- Validation-by-integration tested against analytic ODEs (must flag a
  deliberately corrupted solution).
- Diagnostics fed deliberately broken models (unbounded, infeasible, bad
  dimensions, singular Jacobian) → must emit the right diagnosis.

### External test suites (from arnold-neumaier.at/glopt/test.html)

| Suite | Validates | Role |
|---|---|---|
| Moré/Garbow/Hillstrom, Rosenbrock | `ad/` + unconstrained solve | AD accuracy + basic solve |
| **Hock-Schittkowski** | `solver/` + full chain | Primary end-to-end oracle (published optima) |
| COCONUT / CUTEst | full chain, scale/stress | Later-phase robustness & scaling |
| Space Mission Design (trajectory opt) | `transcription/` + `model/` | Real optimal-control validation |

**Format note:** these suites ship as GAMS/AMPL/SIF/Fortran. v1 does **not**
build parsers — hand-port a curated Hock-Schittkowski subset into the DSL as
fixtures. Parsing CUTEst/COCONUT at scale is a separate later project.

## 7. Using the Framework: Solving a New Problem

Because of the layering, there are **three entry points** depending on how much
structure a problem has. Decision tree:

1. **Trajectory/dynamics problem?** → Option A (declare it via the DSL).
2. **Plain `min f(z) s.t. g(z)`?** → Option B (hand the NLP core your functions).
3. **Needs structure the DSL can't express yet?** → Option C (add one component
   at a seam; core untouched).

For nearly all day-to-day use the user is in Option A.

### Option A — Declare an optimal-control problem (main path)

The user describes structure (states, controls, dynamics, constraints, cost),
never the NLP or any derivative. Queue example:

```cpp
Problem prob;
auto q    = prob.add_state("queue_length");
auto rate = prob.add_control("service_rate");

prob.set_dynamics(q, [](auto& x, auto& u, auto t) {
    return ARRIVAL_RATE - u[service_rate];
});

prob.add_path_constraint(q >= 0.0);
prob.add_path_constraint(0.0 <= rate <= MAX_RATE);
prob.add_boundary_constraint(q.initial() == 10.0);
prob.set_cost(integral(q + WEIGHT * rate * rate));

auto nlp    = HermiteSimpson(num_nodes).compile(prob);
auto result = IpoptAdapter().solve(nlp, sim::linear_guess(prob));
```

The framework discretizes time, builds `z`, writes defect constraints, detects
sparsity, codegens Jacobian/Hessian, and calls the solver.

### Option B — General sparse NLP directly (`nlp/`)

No time/dynamics — just `min f(z) s.t. g(z)`. Skip DSL and transcription; the AD
layer still provides sparse derivatives automatically. This is the path used for
the hand-ported Hock-Schittkowski test problems.

```cpp
NLPProblem nlp(num_vars, num_constraints);
nlp.set_objective([](auto& z) { return ...; });
nlp.set_constraints([](auto& z) { return ...; });
nlp.set_bounds(zL, zU, gL, gU);
auto result = IpoptAdapter().solve(nlp, z0);
```

### Option C — Extend the framework for a new problem class

No rewrite; add a piece at a seam:

| You have… | You implement… | Where |
|---|---|---|
| A new discretization scheme | a `Transcription` subclass | `transcription/` |
| Multiple linked phases | phase objects + linkage constraints | `transcription/` |
| Algebraic (DAE) constraints | algebraic residuals emitted at nodes | `transcription/` |
| A new solver to try | a `Solver` adapter | `solver/` |
| A faster AD backend | an AD backend behind the interface | `ad/` |

## 8. Model Composition (splitting a Problem across files)

Complex models are organized into **`Component`s** (sub-models), each typically
its own file, composed into a parent `Problem`. Each component is understandable
and testable in isolation.

A `Component` may contribute any of:
- **states + dynamics** (a true subsystem),
- **derived quantities** — a named value computed from states/controls/other
  derived values (e.g. `service_rate = g(queue_length, t)`),
- **constraints** and **cost terms**.

The parent wires component inputs/outputs **by name**: e.g. `queue.cpp` declares
the `queue_length` state; `service_model.cpp` declares a component that reads
`queue_length` and publishes `service_rate`; the queue's dynamics consume the
published `service_rate`.

### Two flavors of derived quantity

A derived quantity is declared as one of two kinds; the user chooses per
sub-model:

1. **Inline expression** — substituted directly into consumers' equations. No
   new NLP variable; smallest, sparsest NLP. Best for simple explicit relations.
2. **Algebraic variable** — becomes a real NLP variable with a defining
   constraint (`v − g(x,u,t) = 0`) enforced at each collocation node (DAE-style).
   Larger NLP, but supports implicit/complex relations and makes the quantity
   directly inspectable in the solution. **This reuses the planned DAE seam in
   `transcription/`.**

```cpp
// service_model.cpp
Component service_model() {
    Component c("service");
    auto q = c.input_state("queue_length");        // wired by name from parent

    // Flavor 1: inline expression
    c.add_derived("service_rate", [](auto& x, auto& u, auto t) {
        return BASE_RATE + SLOPE * x[queue_length];
    });

    // Flavor 2 (alternative): algebraic variable with defining relation
    // c.add_algebraic("service_rate", defining_residual, /*bounds*/ 0.0, MAX_RATE);
    return c;
}

// queue.cpp
prob.add_component(service_model());
prob.set_dynamics(q, [](auto& x, auto& u, auto& d, auto t) {
    return ARRIVAL_RATE - d[service_rate];          // consume published quantity
});
```

**Composition contract:** components declare named inputs (states/controls/
derived values they consume) and named outputs (states/derived/algebraic they
publish). The parent resolves the name graph at `build()`, erroring on unresolved
or duplicated names, and orders inline-expression evaluation topologically
(cycles among inline expressions are a setup error; genuine implicit relations
must use the algebraic-variable flavor).

## 9. Build & Tooling

- **CMake** with `FetchContent`/`find_package` for IPOPT, NLopt,
  CppAD/CppADCodeGen, Eigen, GoogleTest.
- **CI** runs the full bottom-up test pyramid.
- FD + complex-step derivative-verification helpers live in a shared test utility.

## 10. Explicitly Out of Scope for v1

- Mesh refinement (adaptive node placement) — later phase.
- Pseudospectral collocation — later phase (interface will allow it).
- AMPL/SIF/CUTEst parsers — later, separate project.
- Enzyme AD backend — interface allows it; not implemented in v1.
- Multi-phase / DAE / hybrid — architecture supports them additively; not v1.
