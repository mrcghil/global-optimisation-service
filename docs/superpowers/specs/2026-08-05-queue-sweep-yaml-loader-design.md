# Queue Sweep YAML Loader + Grouped Axes — Design

Date: 2026-08-05

## Goal

Author parameter sweeps for the queue model in a YAML config and run them
end-to-end to the results dashboard. Add two authoring conveniences the current
spec layer lacks:

1. A `!range [start, stop, count]` YAML tag that expands to a `count`-point
   inclusive linspace.
2. **Grouped axes**: sweep axes organised into groups where axes *within* a
   group vary together (zip, equal-length required) and *across* groups are
   combined by cartesian product (grid).

The first concrete sweep varies both queue-model parameters (`arrival_rate` and
`cost_weight`) as a 3×3 grid (9 runs).

## Context

- **Queue model** (`tools/dashboard_export/gen_sample.cpp`, `tests/spec/queue_fixture.hpp`):
  state `queue_length` (init 10, bounds [0, 1e19]); control `service_rate`
  (bounds [0, 5]); parameters `arrival_rate` (default 2.0, [0, 10]) and
  `cost_weight` (default 0.1, [0, 10]). Dynamics `dq/dt = arrival_rate - service_rate`;
  objective `∫(queue_length + cost_weight · service_rate²)`. Registered as
  `{"queue", "v1"}`.
- **Spec layer** (`include/goss/spec/specs.hpp`): typed
  `RunSpec → SweepSpec → CampaignSpec`, JSON-serializable via `nlohmann::json`
  (`src/spec/json.cpp`). `SweepSpec::expand()` (`src/spec/specs.cpp`) currently
  supports a single flat combinator — `product` or `zip` — over `axes`.
- **Execution** (`include/goss/spec/executor.hpp`): `execute_campaign` →
  `sim::write_campaign` → `sim::export_dashboard_data`. `execute_sweep` calls
  `SweepSpec::expand()` and copies `spec.axes` into the archive for the dashboard
  manifest (`src/spec/executor.cpp:142,166`; `src/sim/archive.cpp:45`).
- **Dashboard** (`dashboard/`): Astro app reads static `index.json` +
  `sweep/<slug>.json` + `run/<id>.json` from a `/data` dir.

**Gap:** there is no YAML support and no config-driven runner. Specs serialize to
JSON only; nothing loads a config from disk (`gen_sample.cpp` hardcodes its
campaign and is not a build target).

## Chosen approach

**Thin loader + minimal spec extension** (Approach A of the brainstorm).

- A new YAML loader resolves `!range` and grouped axes and produces a
  `CampaignSpec`.
- `SweepSpec` gains an optional `groups` field so grouping is first-class and
  flows unchanged to the dashboard. The existing flat `axes` + `combinator` path
  is left intact for backward-compatibility only (see Deprecation).

Rejected: fully expanding groups in the loader into a flat `product` sweep — true
zip-within-group (2+ params changing together) cannot be expressed as one product
combinator, so the loader would have to fake it. Rejected: a YAML→JSON shim with
no `!range`/groups — drops the requested features.

## Components

### 1. YAML loader + CLI

- New static lib target `goss_config`:
  - `include/goss/config/yaml_loader.hpp` — public API
    `CampaignSpec load_campaign_from_yaml(const std::string& path);`
  - `src/config/yaml_loader.cpp` — implementation.
- YAML parser: `yaml-cpp` via `FetchContent` (matches the repo's
  googletest/nlohmann pattern; no system dependency).
- Custom tag on any `values` node:
  - `!range [start, stop, count]` → `count`-point inclusive linspace.
    `count >= 1` required (`SpecError`/loader error otherwise); `count == 1`
    yields `[start]`. Plain YAML lists remain valid explicit values.
  - No `!linspace` alias for now (YAGNI); add later only if requested.
- New CLI `goss_run_sweep <config.yaml> [max_workers]`:
  loads the campaign → registers `{"queue","v1"}` → `execute_campaign` →
  `sim::write_campaign` → `sim::export_dashboard_data` (HDF5-gated).
- The `build_queue` lambda currently duplicated in `gen_sample.cpp` is lifted
  into a shared header (`tools/dashboard_export/queue_model.hpp`) and reused by
  both `gen_sample.cpp` and `goss_run_sweep` so the model definition is not
  duplicated.

### 2. Grouped-axes spec extension

- `specs.hpp`:
  ```cpp
  struct AxisGroup { std::vector<Axis> axes; };  // zip within a group
  ```
  and add `std::vector<AxisGroup> groups;` (default empty) to `SweepSpec`.
- `SweepSpec::expand()`: when `groups` is non-empty, zip each group internally
  (all axes in a group must share length — reuse the existing equal-length zip
  error), then combine groups by cartesian product via the existing
  `sim::make_grid`. When `groups` is empty, behaviour is exactly as today.
- `json.cpp`/`json.hpp`: add `to_json`/`from_json` for `AxisGroup` and the
  `groups` field on `SweepSpec`, reading `groups` with a default so pre-existing
  JSON without the field still parses.
- Dashboard manifest: `execute_sweep` flattens `groups` (all grouped axes, in
  order) into `archive.axes`, so `write_sweep` and `dashboard_export` need zero
  changes.

### 3. The config

`configs/queue_sweep.yaml` (new `configs/` dir):

```yaml
name: queue study
sweeps:
  - label: arrival_x_cost
    base:
      problem: { name: queue, version: v1 }
      parameters: { arrival_rate: 2.0, cost_weight: 0.1 }
      discretization: { scheme: hermite_simpson, t_initial: 0.0, t_final: 5.0, num_intervals: 25 }
      solver: { kind: ipopt }
      storage: { root: "goss-results" }
    groups:
      - [ { parameter: arrival_rate, values: !range [1.0, 3.0, 3] } ]
      - [ { parameter: cost_weight,  values: [0.05, 0.1, 0.5] } ]
```

Two single-axis groups of 3 → 3×3 = 9 runs. Demonstrates both a `!range` axis
and an explicit-list axis. The loader fills spec defaults for omitted fields
(solver tunables, guess, storage flags, etc.).

## Data flow

`queue_sweep.yaml`
→ `load_campaign_from_yaml` (resolves `!range`, builds `groups`)
→ `CampaignSpec`
→ `execute_campaign` (per sweep: `SweepSpec::expand()` zips groups then products them)
→ `CampaignArchive`
→ `sim::write_campaign` (`goss-results/…` manifests + HDF5)
→ `sim::export_dashboard_data` (`data/index.json`, `sweep/<slug>.json`, `run/<id>.json`)
→ Astro dashboard renders the 3×3 grid.

## Error handling

- Loader raises a clear error (`SpecError` or a `config` error type consistent
  with the codebase) on: file not found, malformed YAML, `!range` with
  `count < 1` or wrong arity, non-numeric values.
- `SweepSpec::expand()` throws `SpecError` on unequal-length axes within a group
  (reusing the existing zip message) and on empty axes (existing behaviour).
- `execute_run` continues to report per-point solve failures via
  `RunArchive::result.status` (does not throw), unchanged.

## Backward-compat & deprecation

The following exist **only for backward compatibility** and are candidates for
removal once grouped axes are validated as the superior authoring model. Useless
code should be dropped rather than carried indefinitely.

- **Provisional (to be removed):**
  - `SweepSpec::axes` (flat axis list) and `SweepSpec::combinator`
    (`"product"`/`"zip"`).
  - The legacy branch of `SweepSpec::expand()` that consumes `axes` +
    `combinator`.
  - The `"axes"` and `"combinator"` JSON keys and their (de)serialization in
    `json.cpp`.
- **Replacement:** `SweepSpec::groups` (+ `AxisGroup`) expresses everything the
  flat path did — a single group reproduces `zip`, and single-axis groups
  reproduce `product` — plus mixed zip/product grids the flat path cannot.
- **Removal criteria:** once (a) the loader emits only `groups`, (b) no callers
  or tests rely on the flat path, and (c) the grouped path is exercised
  end-to-end to the dashboard, delete the provisional fields, the legacy
  `expand()` branch, and their JSON keys. The dashboard manifest already sees a
  flat `axes` list (produced by flattening `groups`), so it is unaffected by the
  removal.

These deprecation notes are mirrored as doc comments in `specs.hpp` (on the
`axes`/`combinator` fields) and above the legacy branch in `expand()`.

## Testing

- `tests/config/test_yaml_loader.cpp` (new `goss_config_tests` target):
  - `!range` count, inclusive endpoints, and `count == 1` edge case.
  - Explicit list values.
  - Groups → zip-within, product-across expansion count and values.
  - Unequal-length axes within a group → throws.
  - Round-trip: loaded `CampaignSpec` equals an equivalent hand-built one.
- `tests/spec/test_specs_json.cpp`: `groups` JSON round-trip; legacy JSON without
  `groups` still parses.
- `SweepSpec::expand()` coverage for the grouped path (zip + product counts and
  values), alongside the retained legacy-path tests.

## End-to-end verification

Build `goss_run_sweep`, run it on `configs/queue_sweep.yaml`, confirm
`goss-results/` manifests and dashboard `data/` JSON are produced, then point the
Astro dashboard at the output and confirm the 3×3 grid renders with 9 runs.

## Future scope (not this build)

- A dedicated CLI for defining sweeps with more authoring freedom.
- Server-side sweep creation embedded in the results dashboard (rendered
  server-side, not client-side JS). The current Astro app reads static JSON; this
  would require a server-side endpoint/action.
- Eventual deletion of the provisional flat `axes`/`combinator` path per the
  removal criteria above.
