# Final Fix Report: sim support layer pre-merge wave

**Date:** 2026-07-31
**Branch:** sim-layer
**Full suite:** 97/97 PASSED

---

## Changes Applied

### C1 (Critical) — validate_by_integration: ocp-vs-layout dimension cross-checks

**File:** `include/goss/sim/validation.hpp`

Added three guards immediately after the existing `result.x.size()` guard:
- `ocp.num_states != layout.num_states()` → throws SimError
- `ocp.num_controls != layout.num_controls()` → throws SimError
- `ocp.mesh.num_nodes() != layout.num_nodes()` → throws SimError

These ensure that a mismatched OcpProblem/VariableLayout pair is caught at the entry point rather than silently producing wrong indices or undefined behavior.

---

### I1 (Important) — write_csv: post-write failure check

**File:** `include/goss/sim/trajectory.hpp`

Added `if (!file)` check after `file << to_csv(traj)` to detect silent write failures (e.g. disk full, filesystem errors that only surface after the stream is flushed). Throws `SimError("write_csv: write failed for '...'")`.

---

### M1 (Minor) — initial_guess: replace magic 1e19 with transcription::kInf

**File:** `include/goss/sim/initial_guess.hpp`

- Added `#include "goss/transcription/transcription.hpp"` (defines `kInf = 2e19`)
- Replaced `std::abs(lo) < 1e19 && std::abs(hi) < 1e19` with `std::abs(lo) < transcription::kInf && std::abs(hi) < transcription::kInf`

This makes the finite-bound check consistent with the sentinel value used throughout the transcription layer.

---

### I2 — Diagnostics.NonSuccessIgnoresBadIntegration (new test)

**File:** `tests/sim/test_diagnostics.cpp`

Verifies that a non-Success status (InfeasibleProblem) with a large integration error (999.0) still reports the infeasibility in `d.summary` rather than the re-integration failure. Confirms the diagnose() logic correctly skips integration checks for non-successful solves.

---

### M2 — Diagnostics.NumericalErrorGivesAdvice (new test)

**File:** `tests/sim/test_diagnostics.cpp`

Verifies that `NumericalError` status produces a non-empty `d.advice` string, exercising the NumericalError branch of the diagnose() mapping.

---

### M3 — Validation.ControlledProblemValidates (new test)

**File:** `tests/sim/test_validation.cpp`

Added `MinEnergyDyn` functor (`dx/dt = u`) and a test using:
- State x (position), control u (velocity)
- Boundary conditions: x(0)=0, x(1)=1
- Control bounds: [-10, 10]
- 20 intervals, HermiteSimpson transcription, IpoptSolver, linear_guess

**M3 controlled-validation error observed: 4.440892e-16** (machine epsilon)

The linear dynamics `dx/dt = u` is integrated exactly by RK4 when the control is piecewise-linearly interpolated (the intermediate stages sample the same linear interpolant), so the RK4 solution matches the collocated HS solution to floating-point precision. The threshold of 1e-3 is met with enormous margin, confirming the control sampler works correctly end-to-end.

---

## Test Suite Result

```
97/97 tests passed, 0 tests failed
Total Test time (real) = 34.32 sec
```

New tests: Validation.ControlledProblemValidates (#88), Diagnostics.NonSuccessIgnoresBadIntegration (#95), Diagnostics.NumericalErrorGivesAdvice (#96).
