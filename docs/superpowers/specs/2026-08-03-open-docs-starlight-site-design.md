# Public Documentation Site (`open_docs/`) via Starlight — Design

**Date:** 2026-08-03
**Status:** Approved design, pre-implementation

## 1. Purpose

Build a public-facing documentation site for **goss** (a C++ direct-collocation
trajectory-optimization solver) using [Starlight](https://starlight.astro.build/).
The site targets external readers and must showcase Starlight's features
(i18n-ready, component-rich, auto sidebar). It lives in `open_docs/` and is
distinct from the private `docs/` folder, which holds internal research,
literature review, plans, and specs and is **never** referenced or copied into
the public site.

The public site serves two layered audiences in a single sidebar:

- **Overview track (evaluators):** what goss is, its five-layer architecture,
  transcription schemes, and the validation/benchmark evidence.
- **Hands-on track (library users):** the modeling DSL, the three entry points,
  and worked examples.

### Closed-source constraint

goss is a **closed-source** repository. The public docs MUST NOT include:

- build-from-source instructions,
- dependency-installation steps,
- repository clone/checkout instructions,
- anything that leaks internal build tooling.

"Getting started" is therefore a **conceptual first solve**: annotated example
code that illustrates the public API surface, framed as "here is what a goss
program looks like," not "run this locally."

## 2. Architecture & Isolation

The site is fully self-contained under `open_docs/`:

- Its own `package.json`, `astro.config.mjs`, and `node_modules`.
- No coupling to the C++ CMake build. Building the site never requires a C++
  toolchain, IPOPT, or any goss binary.
- `.gitignore` gains `open_docs/node_modules/` and `open_docs/dist/`.

Tooling: **Astro + `@astrojs/starlight`**. i18n is **configured** with English
as the only current locale, so adding a locale later is a config edit — not a
content restructure.

## 3. Content Structure

```
open_docs/
  package.json
  astro.config.mjs
  .gitignore                        (node_modules/, dist/)
  src/
    content/
      docs/
        en/
          index.mdx                 Landing: hero + CardGrid
          overview/
            what-is-goss.md         Problem class + motivating queue example
            architecture.md         Five layers; the flat-z invariant seam
            transcription-schemes.md Trapezoidal, Hermite-Simpson, hp-pseudospectral
            ad-pipeline.md          CppADCodeGen: record → sparsity → codegen → JIT
          guides/
            first-solve.md          Conceptual walkthrough (queue), <Steps>
            modeling-dsl.md         states/controls/dynamics/constraints/cost
            entry-points.md         DSL / raw NLP / extend-at-a-seam (<Tabs>)
            examples.md             Queue + double-integrator, annotated
          reference/
            validation.md           RK4 validate-by-integration, HS oracle, convergence order
            benchmarks.md           Tables rendered from committed data
    data/
      benchmarks.json               Committed seed data (scheme × solver matrix)
      convergence.json              Committed seed data (convergence orders)
    components/
      BenchmarkTable.astro          Reads benchmarks.json → renders a table
      ConvergenceTable.astro        Reads convergence.json → renders a table
  scripts/
    regen-benchmarks.mjs            Runs goss_bench_export, CSV → JSON into src/data/
    README.md                       How to regenerate (requires a built C++ toolchain)
```

Starlight features exercised: `<Card>`/`<CardGrid>` on the landing page,
`<Tabs>` for the three entry points, `<Steps>` for the first-solve walkthrough,
`<Aside>` callouts, code blocks with titles/filenames, and the auto-generated
sidebar from the `overview` / `guides` / `reference` groups.

### Content sourcing

All prose is authored from the existing internal design
(`docs/superpowers/specs/2026-07-31-nlp-collocation-framework-design.md`) and
the current codebase (`include/goss/**`, `tests/**`). Code snippets in the
guides mirror the DSL surface shown in that design doc (queue example,
double-integrator) and the public headers. No internal-only material (research
strands, preferred-stacks, plans) is surfaced.

## 4. Validation & Benchmarks — Hybrid Data Pipeline

The site renders numbers from **committed** data files; a separate script
regenerates those files from the live C++ harness on demand. The doc build
depends only on the committed JSON, never on the C++ build.

### Committed data

- `src/data/benchmarks.json`: one record per (scheme, solver) with
  `scheme`, `solver`, `status`, `objective`, `elapsed_s`, `validation_error`,
  `num_variables` — mirroring `goss::bench::to_csv`'s columns
  (`include/goss/bench/report.hpp`).
- `src/data/convergence.json`: convergence-order results (scheme, node counts,
  observed error, theoretical rate) drawn from the accuracy test suite
  (`tests/accuracy/test_convergence_order.cpp`).

Both are **seeded now** with real numbers taken from the existing bench/accuracy
tests, so the site renders before anyone runs the regen script.

### Regen wired to the C++ harness

The existing flagship bench test (`tests/bench/test_bench_flagship.cpp`) builds
the scheme × solver result matrix in memory and can serialize it via
`goss::bench::write_csv`, but no standalone binary currently writes it to disk.
The design adds a minimal exporter:

- `tools/bench_export/main.cpp` — a small (~40-line) executable that assembles
  the same flagship (scheme × solver) matrix used by the bench test and calls
  `goss::bench::write_csv(results, out_path)`, where `out_path` is `argv[1]`.
- A CMake target `goss_bench_export` linking `goss_bench` (and its transitive
  deps), added alongside the existing bench target in `CMakeLists.txt`. It is a
  normal executable target, not a test.

The Node regen script:

- `open_docs/scripts/regen-benchmarks.mjs` — runs the already-built
  `goss_bench_export` binary from an existing `build/` directory (path passed as
  an argument or discovered), reads the emitted CSV, converts it to
  `benchmarks.json`, and writes it into `src/data/`. It fails loudly with a
  clear message if the binary or `build/` is missing (it does NOT attempt to
  build the C++ project).
- `open_docs/scripts/README.md` documents the two-step flow: (1) build the C++
  project so `goss_bench_export` exists; (2) run `node scripts/regen-benchmarks.mjs`.

Convergence data regeneration is left as a documented manual step in the same
README for now (the accuracy tests do not yet emit a machine-readable file);
`convergence.json` is maintained by hand until a matching exporter is warranted.

## 5. i18n & Configuration

`astro.config.mjs` configures Starlight with:

- `defaultLocale: 'en'` and `locales: { en: { label: 'English' } }`.
- `title`, `description`, and a `sidebar` with three groups (Overview, Guides,
  Reference) using `autogenerate` per directory.
- Content resolved from `src/content/docs/en/`.

No `base` path and no deploy/hosting configuration — **local build only** for
now. `package.json` provides `dev`, `build`, and `preview` scripts.

## 6. Testing / Verification

- `npm run build` in `open_docs/` succeeds (Astro/Starlight build passes,
  no broken internal links, data components render).
- `npm run dev` serves the site; landing page, all sidebar entries, the Tabs
  entry-points page, and the two data-rendered tables display correctly in a
  browser.
- The `goss_bench_export` target compiles and, when run, writes a CSV whose
  header matches `goss::bench::to_csv`.
- `regen-benchmarks.mjs` run against a built tree reproduces a
  `benchmarks.json` structurally identical to the committed seed.

## 7. Explicitly Out of Scope

- CI, hosting, or deployment (GitHub Pages / Netlify / etc.).
- Any build-from-source, dependency-install, or clone instructions in the
  public docs (closed-source constraint).
- Copying or linking any content from the private `docs/` folder.
- Additional locales beyond English (i18n is configured but single-locale).
- New transcription schemes, solvers, or library features — the C++ change is
  limited to the `goss_bench_export` exporter.
- An automated convergence-data exporter (manual for now).
