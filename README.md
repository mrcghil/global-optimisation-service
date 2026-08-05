# GOSS — Global Optimization Solver

GOSS solves optimal-control problems (OCPs) by direct transcription: it turns a
continuous-time model into a nonlinear program (NLP), solves it with IPOPT or
NLopt, and can sweep a problem across a grid of parameters and publish the
results to a browser dashboard.

The canonical worked example is a **queue model** — minimize
`∫(queue_length + cost_weight · service_rate²)` subject to
`dq/dt = arrival_rate − service_rate` — which appears throughout the tests,
tools, and the `configs/queue_sweep.yaml` demo.

---

## Repository layout

| Path | Contents |
|------|----------|
| `include/goss/` | Public headers, one directory per subsystem (see below). Header-only interfaces live here. |
| `src/` | Implementation `.cpp` files, mirroring the `include/goss/` subsystem tree. |
| `tools/` | Standalone executables: `run_sweep/` (the `goss_run_sweep` CLI), `dashboard_export/` (JSON exporter + `gen_sample`), `bench_export/` (benchmark CSV). |
| `tests/` | GoogleTest suites, one directory per subsystem (`ad/ nlp/ solver/ transcription/ model/ sim/ spec/ config/ bench/ accuracy/`). |
| `configs/` | Runnable YAML sweep/campaign configs, e.g. `queue_sweep.yaml`. |
| `dashboard/` | Astro + D3 app that visualizes solve results (single runs, sweeps, campaigns). See `dashboard/README.md`. |
| `open_docs/` | Separate Astro documentation site (`goss-open-docs`) with benchmark write-ups and literature. |
| `docs/` | Design specs, plans, literature review, and preferred-stack notes (`docs/superpowers/specs`, `docs/superpowers/plans`, `docs/literature-review`, …). |
| `scripts/` | `dev.sh` — run any command inside the build container. |
| `CMakeLists.txt` | Single build definition for every library, test, and tool target. |
| `Dockerfile` / `docker-compose.yml` | The build environment (Ubuntu 24.04 + all native deps) and the CI entrypoint. |
| `build/` | Out-of-source build tree (git-ignored). |
| `goss-results/`, `dashboard-data/` | Generated run output (git-ignored) — created when you run a sweep. |

### Subsystem layers (`include/goss/<name>` ↔ `src/<name>`)

These are internal CMake library targets, layered bottom-up. **They are not git
submodules** — the whole repository is one tree with no `.gitmodules`.

| Subsystem | Library target | Responsibility |
|-----------|----------------|----------------|
| `ad/` | `goss_ad` (interface) + `goss_ad_impl` | Automatic differentiation backend (CppAD / CppADCodeGen JIT). |
| `model/` | `goss_model` (interface) | The model DSL: declare states, controls, parameters, dynamics, and objective. |
| `transcription/` | `goss_transcription` | Direct-transcription schemes: trapezoidal, Hermite–Simpson, Legendre–Gauss–Lobatto, plus meshing. |
| `nlp/` | `goss_nlp` | The NLP problem abstraction the solvers consume. |
| `solver/` | `goss_solver` (+ `goss_ipopt_iface`, `goss_nlopt_iface`) | IPOPT and NLopt wrappers. |
| `sim/` | `goss_sim` (interface) + `goss_sim_impl` | Parameter sweeps (parallel process pool), result archives (HDF5 + JSON), dashboard export. |
| `spec/` | `goss_spec` | Typed `RunSpec`/`SweepSpec`/`CampaignSpec`, the problem registry, JSON (de)serialization, and the executor. |
| `config/` | `goss_config` | YAML loader (`load_campaign_from_yaml`) with the `!range` tag and grouped axes. |
| `bench/` | `goss_bench` (interface) | Benchmark harness and reporting. |

---

## Dependencies

GOSS depends on native C++ libraries that are **not** vendored. The supported
path is to build inside the provided Docker image, which installs them all.

**System libraries (provided by the Docker image):**

- A C++17 compiler, CMake ≥ 3.20
- CppAD + CppADCodeGen (built from source in the image)
- IPOPT and NLopt (found via `pkg-config`)
- LAPACK / BLAS, Eigen3
- libhdf5 (optional — enables the `.h5` run archives and the dashboard exporter;
  without it, GOSS still builds and writes JSON manifests only)

**Fetched automatically at configure time (CMake `FetchContent`, no action needed):**

- GoogleTest `v1.14.0`
- nlohmann/json `v3.11.3`
- yaml-cpp `0.8.0`
- HighFive `v2.9.0` (only when libhdf5 is present)

**Node tooling (only for the two Astro sites):** Node.js + npm for `dashboard/`
and `open_docs/`.

---

## Building and running

Everything C++ builds inside the `goss-build-base` Docker image. There are three
equivalent ways to drive it.

### Option A — `scripts/dev.sh` (recommended for ad-hoc commands)

`dev.sh` builds the image on first use (mounting the repo at `/work`) and runs
whatever command you pass:

```bash
# Configure + build everything
./scripts/dev.sh "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"

# Open an interactive shell in the container
./scripts/dev.sh
```

> On a corporate network that intercepts TLS (Zscaler), the image needs the root
> CA. `dev.sh` defaults `INJECT_ZSCALER_CA=true`; set it to `false` elsewhere.

### Option B — `docker compose` (the CI entrypoint)

Configures, builds, and runs the **entire** test suite in one shot:

```bash
docker compose build
docker compose up          # cmake -S . -B build && cmake --build build && ctest
# On a Zscaler machine: INJECT_ZSCALER_CA=true docker compose build
```

### Option C — raw `docker run`

```bash
docker run --rm -t -v "$PWD:/work" -w /work goss-build-base:local \
  bash -c "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
```

### Building a single target

```bash
./scripts/dev.sh "cmake --build build --target goss_run_sweep -j"
```

Key targets: the libraries above; the tools `goss_run_sweep`,
`goss_dashboard_export`, `goss_bench_export`; and the `*_tests` executables.

---

## Running tests

Tests are GoogleTest binaries registered with CTest. Build them first, then run.

### Run the whole suite (as CI does)

```bash
./scripts/dev.sh "cmake --build build -j && ctest --test-dir build --output-on-failure"
```

### Run one subsystem's tests

Each subsystem has its own test executable
(`goss_ad_tests`, `goss_nlp_tests`, `goss_solver_tests`,
`goss_transcription_tests`, `goss_model_tests`, `goss_sim_tests`,
`goss_spec_tests`, `goss_config_tests`, `goss_bench_tests`,
`goss_accuracy_tests`). Build and run the binary directly:

```bash
./scripts/dev.sh "cmake --build build --target goss_config_tests -j && ./build/goss_config_tests"
```

> **Gotcha:** `ctest -R goss_config_tests` finds **no** tests. `gtest_discover_tests`
> registers individual *test names* (e.g. `YamlLoader.RangeTagExpandsInclusiveLinspace`),
> not target names. To filter, either run the binary with
> `--gtest_filter='YamlLoader*'`, or use `ctest -R` with an actual test-name pattern.

Some suites (`goss_spec_tests`, `goss_accuracy_tests`) JIT-compile problems and
take ~1 minute — that is expected.

---

## Running a parameter sweep → dashboard

The end-to-end flow: a YAML config → solved grid → JSON contract → browser.

1. **Build the runner:**

   ```bash
   ./scripts/dev.sh "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target goss_run_sweep -j"
   ```

2. **Run the demo sweep** (a 3×3 grid over `arrival_rate` × `cost_weight`):

   ```bash
   ./scripts/dev.sh "./build/goss_run_sweep configs/queue_sweep.yaml dashboard-data 4"
   ```

   This writes result manifests under `goss-results/` and, when HDF5 is enabled,
   the dashboard JSON contract under `dashboard-data/` (`index.json`,
   `sweep/<slug>.json`, `run/<id>.json`).

3. **View it in the dashboard:**

   ```bash
   cp -R dashboard-data dashboard/public/data     # feed your fresh results in
   cd dashboard && ./node_modules/.bin/astro dev   # http://localhost:4321
   ```

   > Use the `astro` binary directly to serve your own data. `npm run dev` runs a
   > `predev` hook that overwrites `public/data` with the bundled sample set.

### Authoring a sweep config

`configs/queue_sweep.yaml` shows the format. Axes are organized into **groups**:
axes *within* a group advance together (zip, equal length required); *across*
groups they form a cartesian product (grid). Values can be an explicit list or
the `!range [start, stop, count]` tag (an inclusive linspace of `count` points):

```yaml
name: queue study
sweeps:
  - label: arrival_x_cost
    base:
      problem: { name: queue, version: v1 }
      parameters: { arrival_rate: 2.0, cost_weight: 0.1 }
      discretization: { scheme: hermite_simpson, t_final: 5.0, num_intervals: 25 }
      solver: { kind: ipopt }
      storage: { root: "goss-results" }
    groups:
      - [ { parameter: arrival_rate, values: !range [1.0, 3.0, 3] } ]
      - [ { parameter: cost_weight,  values: [0.05, 0.1, 0.5] } ]
```

---

## The two web apps

Both are Astro sites with their own `package.json`. Install deps with `npm install`
in each, then:

- **`dashboard/`** — results viewer. `npm run dev` (serves bundled sample data),
  or serve live results as shown above. Details in `dashboard/README.md`.
- **`open_docs/`** — documentation & benchmark site. `npm run dev`;
  `npm run regen:benchmarks` refreshes benchmark data.
