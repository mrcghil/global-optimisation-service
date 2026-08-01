# Final Fix Report — Benchmark Harness (2026-08-01)

## Summary

Four minor localized fixes applied before merge. Full suite: **120/120 passed**.

---

## m1: CMakeLists.txt — Remove duplicate `goss_solver` in `goss_bench_tests`

**File:** `CMakeLists.txt` (lines 189–193)

`goss_solver` appeared twice in `target_link_libraries` for `goss_bench_tests`:
once explicitly at the start of the list and again after `goss_ad_impl`. The second
occurrence was removed. The resulting list is:

```
goss_bench goss_sim goss_solver goss_model goss_transcription
goss_nlp goss_ad goss_ad_impl
goss_ipopt_iface goss_nlopt_iface cppadcg
$<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}> GTest::gtest_main
```

CMake deduplicates link libraries at the linker level, so this was not causing a
build failure, but the redundancy was confusing and a maintenance hazard.

---

## m2: report.hpp — Separator width `+8` removed and comment added

**File:** `include/goss/bench/report.hpp` (`row_separator` lambda in `to_table`)

**What the +8 was:** The separator line used
`kColScheme + kColSolver + kColStatus + kColObjective + kColTime + kColValErr + kColNVars + 8`
as its dash count. The column widths sum to:
18 + 16 + 20 + 14 + 12 + 16 + 8 = **104**.

The data rows use `std::left` with `std::setw` for each column and no inter-column
separators, pipes, or padding characters between columns. The actual data-row width
is therefore exactly 104 characters. The `+8` was **spurious** — it produced a
separator 8 characters wider than every data row, creating a visual overhang.

**Resolution:** Removed the `+8` so the separator is flush with the data rows
(104 dashes). Added a brief comment explaining the intentional match:

```cpp
// Width is exactly the sum of all column widths: no inter-column separators,
// so the separator is flush with the data rows.
```

No test asserted on separator length (all `to_table` tests check for presence of
strings or newline count, not character width), so all 120 tests continue to pass.

---

## m3: report.hpp — `file.flush()` before post-write error check in `write_csv`

**File:** `include/goss/bench/report.hpp` (`write_csv` function)

Added `file.flush()` between the write and the `if (!file)` check:

```cpp
file << to_csv(results);
file.flush();
if (!file) throw BenchError("write_csv: write failed for '" + path + "'");
```

Without the explicit flush, the `ofstream` destructor would flush on close, but a
buffered write failure would not be visible to the `!file` check at this point in
the function — a silent data-loss risk. The flush forces any OS-level write error
to set the stream's `failbit` before the guard runs.

---

## m4: harness.hpp — Documenting comment before `validate_by_integration` guard

**File:** `include/goss/bench/harness.hpp` (`run_scheme` function)

Added a block comment immediately before the
`if (solve_result.status == goss::solver::SolverStatus::Success)` guard that wraps
the `validate_by_integration` call, explaining the intentional design:

```cpp
// Precondition: a well-behaved solver returns result.x sized to
// layout.total_variables() on Success. If a solver adapter violates that,
// validate_by_integration throws SimError, which propagates out of run_scheme
// — intentional: it surfaces a solver-adapter bug rather than hiding it.
```

No logic was changed.

---

## Test Results

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

**100% tests passed, 0 tests failed out of 120**
Total test time: 36.09 sec
