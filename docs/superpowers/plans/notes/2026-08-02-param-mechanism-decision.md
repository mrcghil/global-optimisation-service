# Parameter Injection Mechanism Decision

**Date:** 2026-08-02
**Task:** Spike — Task 1 of the parameter-binding plan
**Status:** DECIDED — fallback mechanism (pinned decision variables)

---

## Mechanism Chosen: Pinned Decision Variables

**Decision:** Append parameters as extra independent variables in the combined
vector `z = [x..., p...]`. "Injecting" a parameter means supplying its value in
the corresponding tail slot of the `x` vector at every evaluation call.  The
solver layer enforces `lower == upper` bounds on those slots so the optimizer
cannot move them.

---

## Why the Primary (CppAD dynamic parameters) Was Rejected

The primary mechanism required calling `model->new_dynamic(p)` on a
`CppAD::cg::GenericModel<double>` before each `ForwardZero` /
`SparseJacobian` / `SparseHessian` call.

Investigation of `/usr/local/include/cppad/cg/model/generic_model.hpp`
(the pinned CppADCodeGen version installed in the container) confirmed that
`GenericModel` does **not** expose a `new_dynamic()` method.

The `new_dynamic()` function exists on `CppAD::ADFun<Base>` in the core CppAD
library, but `ADFun` is discarded after
`DynamicModelLibraryProcessor::createDynamicLibrary()` runs — the generated C
source encodes parameter values at code-generation time.  There is no runtime
injection hook in the compiled `.so` artifact returned as `GenericModel`.

---

## Exact API Calls Proved by the Spike

### Recording (once per model)

```cpp
// z = [x0, ..., x_{n-1}, p0, ..., p_{m-1}]
const std::size_t num_total = num_vars + num_params;
std::vector<ADCG> az(num_total);
CppAD::Independent(az);           // single-argument form; no dynamic vector
std::vector<ADCG> ay = f(az);     // build expression using az[0..n-1] as vars,
                                  // az[n..n+m-1] as params
CppAD::ADFun<CGD> fun(az, ay);
fun.optimize();
```

### JIT compilation (once per model)

Standard `compile_and_load` pipeline — unchanged from `cppadcg_backend.cpp`:

```cpp
CppAD::cg::ModelCSourceGen<double> source_gen(fun, model_name);
source_gen.setCreateForwardZero(true);
source_gen.setCreateSparseJacobian(true);
// ... other setCreate* calls as needed
CppAD::cg::ModelLibraryCSourceGen<double> library_gen(source_gen);
CppAD::cg::GccCompiler<double> compiler;
CppAD::cg::DynamicModelLibraryProcessor<double> processor(library_gen, model_name);
auto library = processor.createDynamicLibrary(compiler);
auto model   = library->model(model_name);
```

### Parameter injection (per evaluation call)

```cpp
// Build combined vector: decision variables then parameter values
std::vector<double> z(num_vars + num_params);
// fill z[0..num_vars-1] with current decision variable values
// fill z[num_vars..] with current parameter values (the "injection")
z[num_vars + 0] = p[0];  // p0

// Evaluate — same GenericModel call as before, no extra API needed
std::vector<double> y   = model->ForwardZero(z);
std::vector<double> jac = /* model->SparseJacobian(...) */;
```

Note: the Jacobian returned by `SparseJacobian` covers columns for **both**
decision variables and parameter slots.  The caller must select only the
`[0..num_vars-1]` columns as the "true" Jacobian w.r.t. decisions; the
parameter-column entries give sensitivities w.r.t. parameter values, which are
informational.

---

## Spike Validation Results

Test: `DynamicParamSpike.ValueAndJacobianTrackInjectedParameterWithoutRecompile`
- `z = [2.0, 3.0]` → `f = 3 * 4 = 12.0` ✓
- `z = [2.0, 5.0]` → `f = 5 * 4 = 20.0` ✓
- `df/dx0` at `p0=5, x0=2` → `2*5*2 = 20.0` ✓ (Jacobian also reflects injected p)
- Same compiled model used for both evaluations — no re-recording, no re-compile.

CTest result: **1/1 PASSED, 0.11 s**

---

## Impact on Task 2

Task 2 wraps this mechanism in the `CppADCGBackend` public interface:

- `num_parameters()` → returns the count of appended parameter slots (tail of `z`).
- `set_parameters(p)` → stores `p`; the backend inserts values into the
  `z`-vector tail before every `eval()` / `jacobian()` / `hessian()` call.
- The upper-layer abstraction is identical regardless of mechanism; Task 2
  implements it using the pinned-variable slot approach documented above.
