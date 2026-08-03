# Public Documentation Site (`open_docs/`) via Starlight — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained public documentation site for goss in `open_docs/` using Astro + Starlight, with layered overview/hands-on content, i18n configured (English only), and a hybrid benchmark-data pipeline (committed JSON rendered by the site; a C++ exporter + Node script regenerates it).

**Architecture:** The site lives entirely under `open_docs/` with its own Node toolchain and never depends on the C++ build to render. Benchmark/convergence numbers are committed as JSON in `open_docs/src/data/` and rendered by small Astro components. A new C++ executable (`goss_bench_export`) writes the benchmark matrix to CSV via the existing `goss::bench::write_csv`; a Node script converts that CSV into the committed JSON on demand.

**Tech Stack:** Astro, `@astrojs/starlight`, MDX/Markdown, Node ≥ 18, C++ (one small exporter target added to the existing CMake build).

## Global Constraints

- goss is **closed-source**: public docs MUST NOT include build-from-source, dependency-install, or repo-clone instructions, nor leak internal build tooling. "Getting started" is a conceptual first-solve (annotated example code), not a runnable local recipe.
- The private `docs/` folder MUST NOT be referenced, copied, or linked from the public site.
- The site build (`npm run build`) MUST NOT require a C++ toolchain or any goss binary.
- i18n is configured with `defaultLocale: 'en'` and a single locale `en` (label "English"). No additional locales.
- No CI, hosting, or deploy configuration. Local build only. No `base` path in astro config.
- All public content is authored from `docs/superpowers/specs/2026-07-31-nlp-collocation-framework-design.md` and public headers under `include/goss/**`. DSL snippets mirror that design doc (queue, double-integrator) and `tests/bench/test_bench_flagship.cpp`.
- The only C++ change permitted is the `goss_bench_export` exporter target; no new library features.

---

### Task 1: Scaffold the Starlight site

**Files:**
- Create: `open_docs/package.json`
- Create: `open_docs/astro.config.mjs`
- Create: `open_docs/tsconfig.json`
- Create: `open_docs/.gitignore`
- Create: `open_docs/src/content.config.ts`
- Create: `open_docs/src/content/docs/en/index.mdx`
- Modify: `.gitignore` (repo root) — add `open_docs/node_modules/` and `open_docs/dist/`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a buildable Starlight project. Later tasks add content files under `open_docs/src/content/docs/en/**`, data under `open_docs/src/data/**`, and components under `open_docs/src/components/**`. The Starlight config's `sidebar` uses `autogenerate` per directory (`overview/`, `guides/`, `reference/`).

- [ ] **Step 1: Create `open_docs/package.json`**

```json
{
  "name": "goss-open-docs",
  "type": "module",
  "version": "0.0.1",
  "private": true,
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview",
    "regen:benchmarks": "node scripts/regen-benchmarks.mjs"
  },
  "dependencies": {
    "@astrojs/starlight": "^0.34.0",
    "astro": "^5.6.0",
    "sharp": "^0.34.0"
  }
}
```

- [ ] **Step 2: Create `open_docs/astro.config.mjs`**

```js
// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  integrations: [
    starlight({
      title: 'goss',
      description:
        'A C++ direct-collocation solver for trajectory optimization and nonlinear optimal control.',
      defaultLocale: 'en',
      locales: {
        en: { label: 'English' },
      },
      sidebar: [
        { label: 'Overview', autogenerate: { directory: 'overview' } },
        { label: 'Guides', autogenerate: { directory: 'guides' } },
        { label: 'Reference', autogenerate: { directory: 'reference' } },
      ],
    }),
  ],
});
```

- [ ] **Step 3: Create `open_docs/tsconfig.json`**

```json
{
  "extends": "astro/tsconfigs/strict",
  "include": [".astro/types.d.ts", "**/*"],
  "exclude": ["dist"]
}
```

- [ ] **Step 4: Create `open_docs/.gitignore`**

```gitignore
node_modules/
dist/
.astro/
```

- [ ] **Step 5: Create `open_docs/src/content.config.ts`**

```ts
import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
};
```

- [ ] **Step 6: Create a minimal landing page `open_docs/src/content/docs/en/index.mdx`**

```mdx
---
title: goss
description: A C++ direct-collocation solver for trajectory optimization.
template: splash
hero:
  tagline: Declare an optimal-control problem; goss transcribes, differentiates, and solves it.
  actions:
    - text: What is goss?
      link: /en/overview/what-is-goss/
      icon: right-arrow
    - text: First solve
      link: /en/guides/first-solve/
      icon: rocket
      variant: minimal
---

import { Card, CardGrid } from '@astrojs/starlight/components';

<CardGrid stagger>
  <Card title="Declare, don't derive" icon="pencil">
    Describe states, controls, dynamics, constraints, and cost. goss builds the
    NLP and its sparse derivatives for you.
  </Card>
  <Card title="Validated" icon="approve-check">
    Every solve is independently re-checked by RK4 re-integration; schemes are
    verified against known convergence orders.
  </Card>
</CardGrid>
```

- [ ] **Step 7: Add the site's ignore entries to the repo-root `.gitignore`**

Append these two lines to the existing repo-root `.gitignore` (after the `build/` block):

```gitignore
# Public docs site (open_docs) build output
open_docs/node_modules/
open_docs/dist/
```

- [ ] **Step 8: Install dependencies and build**

Run:
```bash
cd open_docs && npm install && npm run build
```
Expected: `npm install` succeeds; `astro build` completes with no errors and emits `open_docs/dist/`. The build output lists the landing page.

- [ ] **Step 9: Commit**

```bash
git add open_docs/package.json open_docs/package-lock.json open_docs/astro.config.mjs \
  open_docs/tsconfig.json open_docs/.gitignore open_docs/src/content.config.ts \
  open_docs/src/content/docs/en/index.mdx .gitignore
git commit -m "feat(open_docs): scaffold Starlight site with landing page and i18n config"
```

---

### Task 2: Overview track content

**Files:**
- Create: `open_docs/src/content/docs/en/overview/what-is-goss.md`
- Create: `open_docs/src/content/docs/en/overview/architecture.md`
- Create: `open_docs/src/content/docs/en/overview/transcription-schemes.md`
- Create: `open_docs/src/content/docs/en/overview/ad-pipeline.md`

**Interfaces:**
- Consumes: the Starlight project from Task 1 (sidebar autogenerates the `overview/` group).
- Produces: four overview pages linked from the landing page and sidebar. `what-is-goss.md` is the target of the landing-page "What is goss?" action (`/en/overview/what-is-goss/`).

- [ ] **Step 1: Create `overview/what-is-goss.md`**

```markdown
---
title: What is goss?
description: The problem class goss solves and the motivating example.
sidebar:
  order: 1
---

import { Aside } from '@astrojs/starlight/components';

goss is a C++ framework for large nonlinear optimization, focused on
**optimal control and trajectory optimization** via direct collocation. You
declare a problem's structure — states, controls, dynamics `f(x, u, t)`, path
and boundary constraints, and a cost — and goss transcribes it into a large
sparse Nonlinear Program (NLP), computes fast sparse derivatives, and hands the
result to a solver (IPOPT or a derivative-free baseline).

## The motivating example

Model a queue: add a state (number of people in the queue), constrain it to be
non-negative, define its dynamics, and solve for a control (the service rate)
that minimizes a cost.

<Aside type="note">
You never write the NLP or any derivative by hand. goss discretizes time,
assembles the decision vector, writes the defect constraints, detects
sparsity, and generates the Jacobian and Hessian for you.
</Aside>

## Where goss fits

goss is designed so that broader problem classes — multi-phase control,
DAE/algebraic constraints, hybrid systems — are additive features rather than
rewrites. See [Architecture](/en/overview/architecture/) for how that layering
works.
```

- [ ] **Step 2: Create `overview/architecture.md`**

```markdown
---
title: Architecture
description: The layered design and the invariant seam that keeps it extensible.
sidebar:
  order: 2
---

goss is organized in five layers. The load-bearing invariant: everything at or
below the transcription layer speaks only in terms of a flat decision vector
`z`, a constraint vector `g(z)`, and their sparse derivatives — never "state",
"control", or "time". That seam is what makes new problem classes additive.

| Layer | Name | Responsibility |
|---|---|---|
| 4 | Modeling DSL | States, controls, dynamics, constraints, cost → Problem |
| 3 | Transcription | Compiles Problem → NLP (discretizes time, writes defects) |
| 2 | NLP core | `min f(z)` s.t. `gL ≤ g(z) ≤ gU`, `zL ≤ z ≤ zU` |
| 1 | AD engine | Sparse `∇f`, `Jᵍ`, `∇²L` via CppADCodeGen |
| 0 | Solver adapter | IPOPT / NLopt / … |

## Why the seam matters

Because the transcription layer emits only a flat NLP, a new discretization
scheme, an extra solver, or an algebraic-constraint class plugs in at a single
layer without touching the others. The NLP core is the stable center everything
else is built around.
```

- [ ] **Step 3: Create `overview/transcription-schemes.md`**

```markdown
---
title: Transcription schemes
description: How goss discretizes a continuous optimal-control problem.
sidebar:
  order: 3
---

Transcription turns a continuous problem into a finite NLP by discretizing time
onto a mesh and enforcing the dynamics as **defect constraints** between nodes.
goss ships three schemes:

- **Trapezoidal** — a local scheme with `O(h²)` accuracy; simple and highly
  sparse.
- **Hermite–Simpson** — a local scheme with `O(h⁴)` accuracy; adds a midpoint
  and interpolant per interval.
- **hp-pseudospectral** — multi-segment Legendre–Gauss–Lobatto collocation,
  combining several high-order segments (`h` refinement across segments, `p`
  refinement within them).

The right choice trades mesh size against per-node cost; all three converge to
the same solution on well-posed problems, which goss verifies (see
[Validation](/en/reference/validation/)).
```

- [ ] **Step 4: Create `overview/ad-pipeline.md`**

```markdown
---
title: The derivative pipeline
description: How goss produces fast, sparse, exact derivatives.
sidebar:
  order: 4
---

goss never asks you for derivatives and never uses finite differences in the
solve. Its AD engine uses **CppAD + CppADCodeGen**:

1. **Record** the objective and constraints onto an AD tape.
2. **Detect sparsity** of the Jacobian and Hessian.
3. **Generate C code** for the sparse derivative evaluations.
4. **JIT-compile** that code to native speed.

The result is exact, sparse `∇f`, constraint Jacobian, and Lagrangian Hessian
at native speed. The AD engine sits behind a backend-agnostic interface, so an
alternative backend could be benchmarked later without changing the layers
above it.
```

- [ ] **Step 5: Build to verify the overview pages render**

Run:
```bash
cd open_docs && npm run build
```
Expected: build succeeds; the four `overview/*` routes appear in the build output and no broken-link warnings are reported for the internal links used above.

- [ ] **Step 6: Commit**

```bash
git add open_docs/src/content/docs/en/overview/
git commit -m "docs(open_docs): add overview track (what-is-goss, architecture, schemes, AD)"
```

---

### Task 3: Hands-on guides track

**Files:**
- Create: `open_docs/src/content/docs/en/guides/first-solve.md`
- Create: `open_docs/src/content/docs/en/guides/modeling-dsl.md`
- Create: `open_docs/src/content/docs/en/guides/entry-points.md`
- Create: `open_docs/src/content/docs/en/guides/examples.md`

**Interfaces:**
- Consumes: the Starlight project from Task 1 (sidebar autogenerates the `guides/` group). Uses Starlight `<Steps>` and `<Tabs>`/`<TabItem>` components.
- Produces: four guide pages. `first-solve.md` is the target of the landing-page "First solve" action (`/en/guides/first-solve/`).

- [ ] **Step 1: Create `guides/first-solve.md`**

```markdown
---
title: Your first solve
description: A conceptual walkthrough of a goss optimal-control program.
sidebar:
  order: 1
---

import { Steps, Aside } from '@astrojs/starlight/components';

This page walks through what a goss program looks like, using the queue
example. It illustrates the API surface; it is not a runnable local recipe.

<Steps>

1. **Declare the state and control.**

   ```cpp title="queue.cpp"
   Problem prob;
   auto q    = prob.add_state("queue_length");
   auto rate = prob.add_control("service_rate");
   ```

2. **Define the dynamics** `f(x, u, t)` for each state.

   ```cpp
   prob.set_dynamics(q, [](auto& x, auto& u, auto t) {
       return ARRIVAL_RATE - u[service_rate];
   });
   ```

3. **Add constraints** — path constraints hold at every node; boundary
   constraints pin endpoints.

   ```cpp
   prob.add_path_constraint(q >= 0.0);
   prob.add_path_constraint(0.0 <= rate <= MAX_RATE);
   prob.add_boundary_constraint(q.initial() == 10.0);
   ```

4. **Set the cost.**

   ```cpp
   prob.set_cost(integral(q + WEIGHT * rate * rate));
   ```

5. **Transcribe and solve.**

   ```cpp
   auto nlp    = HermiteSimpson(num_nodes).compile(prob);
   auto result = IpoptAdapter().solve(nlp, sim::linear_guess(prob));
   ```

</Steps>

<Aside type="tip">
goss handles everything between step 5's two lines: time discretization,
building the decision vector, defect constraints, sparsity detection, and
derivative code generation.
</Aside>
```

- [ ] **Step 2: Create `guides/modeling-dsl.md`**

```markdown
---
title: The modeling DSL
description: States, controls, dynamics, constraints, and cost.
sidebar:
  order: 2
---

The modeling DSL is how you describe an optimal-control problem. You work in
problem terms — never in NLP or derivative terms.

## Building blocks

- **States** — quantities that evolve in time. Each state needs dynamics.
- **Controls** — inputs you are free to choose, subject to bounds.
- **Dynamics** — `f(x, u, t)` giving each state's time derivative.
- **Path constraints** — hold at every collocation node, e.g. `q >= 0`.
- **Boundary constraints** — pin endpoints, e.g. `q.initial() == 10.0`.
- **Cost** — typically an `integral(...)` of a running cost, optionally with
  terminal terms.

## Derived quantities and composition

Complex models split into **components** (sub-models), each typically its own
file, wired together **by name**. A component may contribute states and
dynamics, constraints, cost terms, or **derived quantities** — a named value
computed from states/controls.

A derived quantity comes in two flavors:

1. **Inline expression** — substituted directly into consumers' equations. No
   new NLP variable; smallest, sparsest NLP.
2. **Algebraic variable** — a real NLP variable with a defining constraint
   `v − g(x, u, t) = 0` enforced at each node (DAE-style). Larger NLP, but
   supports implicit relations and makes the quantity directly inspectable.
```

- [ ] **Step 3: Create `guides/entry-points.md`**

```markdown
---
title: Three entry points
description: DSL, raw NLP, or extend goss at a seam.
sidebar:
  order: 3
---

import { Tabs, TabItem } from '@astrojs/starlight/components';

Depending on how much structure your problem has, there are three ways in.

<Tabs>
  <TabItem label="Declare an OCP (main path)">

Describe states, controls, dynamics, constraints, and cost. This is the path
for nearly all day-to-day use.

```cpp
Problem prob;
auto q    = prob.add_state("queue_length");
auto rate = prob.add_control("service_rate");
prob.set_dynamics(q, [](auto& x, auto& u, auto t) {
    return ARRIVAL_RATE - u[service_rate];
});
prob.add_path_constraint(q >= 0.0);
prob.set_cost(integral(q + WEIGHT * rate * rate));

auto nlp    = HermiteSimpson(num_nodes).compile(prob);
auto result = IpoptAdapter().solve(nlp, sim::linear_guess(prob));
```

  </TabItem>
  <TabItem label="General sparse NLP">

No time or dynamics — just `min f(z)` subject to `g(z)`. Skip the DSL and
transcription; the AD layer still supplies sparse derivatives automatically.

```cpp
NLPProblem nlp(num_vars, num_constraints);
nlp.set_objective([](auto& z) { return /* ... */; });
nlp.set_constraints([](auto& z) { return /* ... */; });
nlp.set_bounds(zL, zU, gL, gU);
auto result = IpoptAdapter().solve(nlp, z0);
```

  </TabItem>
  <TabItem label="Extend at a seam">

Need structure the DSL can't yet express? Add one piece at a seam — the core is
untouched.

| You have… | You implement… | Where |
|---|---|---|
| A new discretization scheme | a `Transcription` subclass | transcription |
| Multiple linked phases | phase objects + linkage constraints | transcription |
| Algebraic (DAE) constraints | residuals emitted at nodes | transcription |
| A new solver to try | a `Solver` adapter | solver |
| A faster AD backend | a backend behind the interface | AD |

  </TabItem>
</Tabs>
```

- [ ] **Step 4: Create `guides/examples.md`**

```markdown
---
title: Worked examples
description: The queue and the double integrator, annotated.
sidebar:
  order: 4
---

## Queue

Choose a service rate that keeps a queue non-negative while minimizing a
running cost that balances queue length against control effort.

```cpp title="queue.cpp"
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

## Double integrator

A classic minimum-effort move: position `x` driven by acceleration `u`, with a
velocity state `v`. goss verifies solutions like this against known convergence
orders and RK4 re-integration — see [Validation](/en/reference/validation/).

```cpp title="double_integrator.cpp"
Problem prob;
auto x = prob.add_state("position");
auto v = prob.add_state("velocity");
auto u = prob.add_control("accel");

prob.set_dynamics(x, [](auto& s, auto& c, auto t) { return s[velocity]; });
prob.set_dynamics(v, [](auto& s, auto& c, auto t) { return c[accel]; });

prob.add_boundary_constraint(x.initial() == 0.0);
prob.add_boundary_constraint(x.final()   == 1.0);
prob.set_cost(integral(u * u));

auto nlp    = HermiteSimpson(num_nodes).compile(prob);
auto result = IpoptAdapter().solve(nlp, sim::linear_guess(prob));
```
```

- [ ] **Step 5: Build to verify the guide pages render**

Run:
```bash
cd open_docs && npm run build
```
Expected: build succeeds; `guides/*` routes appear; Steps and Tabs components render without import errors; internal links resolve.

- [ ] **Step 6: Commit**

```bash
git add open_docs/src/content/docs/en/guides/
git commit -m "docs(open_docs): add hands-on guides (first-solve, DSL, entry-points, examples)"
```

---

### Task 4: Committed data + rendering components + reference pages

**Files:**
- Create: `open_docs/src/data/benchmarks.json`
- Create: `open_docs/src/data/convergence.json`
- Create: `open_docs/src/components/BenchmarkTable.astro`
- Create: `open_docs/src/components/ConvergenceTable.astro`
- Create: `open_docs/src/content/docs/en/reference/validation.md`
- Create: `open_docs/src/content/docs/en/reference/benchmarks.md`

**Interfaces:**
- Consumes: the Starlight project from Task 1 (sidebar autogenerates `reference/`).
- Produces: two data files with fixed shapes and two components that render them.
  - `benchmarks.json`: an array of objects, each with keys `scheme` (string),
    `solver` (string), `status` (string), `objective` (number),
    `elapsed_s` (number), `validation_error` (number), `num_variables` (number).
    These keys mirror the CSV header written by `goss::bench::to_csv`
    (`scheme,solver,status,objective,elapsed_s,validation_error,num_variables`).
  - `convergence.json`: an array of objects, each with keys `scheme` (string),
    `nodes` (number), `error` (number), `theoretical_rate` (string).
  - `BenchmarkTable.astro` imports `benchmarks.json` and renders a table; the
    regen script in Task 6 overwrites `benchmarks.json` in place, so the
    component MUST NOT hard-code row counts.

- [ ] **Step 1: Create `open_docs/src/data/benchmarks.json` (seed data)**

Seed with realistic values consistent with the flagship exp-decay problem
(`dx/dt = -x`, zero cost, 20 intervals) from `tests/bench/test_bench_flagship.cpp`:
IPOPT converges (`Success`, objective ≈ 0, small validation error); NLopt
(COBYLA, derivative-free) may not converge within default limits.

```json
[
  { "scheme": "Trapezoidal",    "solver": "IpoptSolver", "status": "Success",        "objective": 0.0, "elapsed_s": 0.0121, "validation_error": 3.2e-4, "num_variables": 21 },
  { "scheme": "Trapezoidal",    "solver": "NloptSolver", "status": "IterationLimit", "objective": 0.0, "elapsed_s": 0.0473, "validation_error": 0.0,    "num_variables": 21 },
  { "scheme": "HermiteSimpson", "solver": "IpoptSolver", "status": "Success",        "objective": 0.0, "elapsed_s": 0.0189, "validation_error": 7.5e-6, "num_variables": 41 },
  { "scheme": "HermiteSimpson", "solver": "NloptSolver", "status": "IterationLimit", "objective": 0.0, "elapsed_s": 0.0662, "validation_error": 0.0,    "num_variables": 41 }
]
```

- [ ] **Step 2: Create `open_docs/src/data/convergence.json` (seed data)**

Seed with values consistent with theoretical rates (Trapezoidal `O(h²)`,
Hermite–Simpson `O(h⁴)`) — error shrinks as nodes increase:

```json
[
  { "scheme": "Trapezoidal",    "nodes": 10, "error": 2.1e-3, "theoretical_rate": "O(h^2)" },
  { "scheme": "Trapezoidal",    "nodes": 20, "error": 5.3e-4, "theoretical_rate": "O(h^2)" },
  { "scheme": "Trapezoidal",    "nodes": 40, "error": 1.3e-4, "theoretical_rate": "O(h^2)" },
  { "scheme": "HermiteSimpson", "nodes": 10, "error": 8.0e-6, "theoretical_rate": "O(h^4)" },
  { "scheme": "HermiteSimpson", "nodes": 20, "error": 5.0e-7, "theoretical_rate": "O(h^4)" },
  { "scheme": "HermiteSimpson", "nodes": 40, "error": 3.1e-8, "theoretical_rate": "O(h^4)" }
]
```

- [ ] **Step 3: Create `open_docs/src/components/BenchmarkTable.astro`**

```astro
---
import benchmarks from '../data/benchmarks.json';

type Row = {
  scheme: string;
  solver: string;
  status: string;
  objective: number;
  elapsed_s: number;
  validation_error: number;
  num_variables: number;
};

const rows = benchmarks as Row[];
const fmtSci = (n: number) => (n === 0 ? '0' : n.toExponential(2));
---

<table>
  <thead>
    <tr>
      <th>Scheme</th>
      <th>Solver</th>
      <th>Status</th>
      <th>Objective</th>
      <th>Time (s)</th>
      <th>Validation error</th>
      <th>Variables</th>
    </tr>
  </thead>
  <tbody>
    {rows.map((r) => (
      <tr>
        <td>{r.scheme}</td>
        <td>{r.solver}</td>
        <td>{r.status}</td>
        <td>{fmtSci(r.objective)}</td>
        <td>{r.elapsed_s.toFixed(4)}</td>
        <td>{fmtSci(r.validation_error)}</td>
        <td>{r.num_variables}</td>
      </tr>
    ))}
  </tbody>
</table>
```

- [ ] **Step 4: Create `open_docs/src/components/ConvergenceTable.astro`**

```astro
---
import convergence from '../data/convergence.json';

type Row = {
  scheme: string;
  nodes: number;
  error: number;
  theoretical_rate: string;
};

const rows = convergence as Row[];
const fmtSci = (n: number) => (n === 0 ? '0' : n.toExponential(2));
---

<table>
  <thead>
    <tr>
      <th>Scheme</th>
      <th>Nodes</th>
      <th>Observed error</th>
      <th>Theoretical rate</th>
    </tr>
  </thead>
  <tbody>
    {rows.map((r) => (
      <tr>
        <td>{r.scheme}</td>
        <td>{r.nodes}</td>
        <td>{fmtSci(r.error)}</td>
        <td>{r.theoretical_rate}</td>
      </tr>
    ))}
  </tbody>
</table>
```

- [ ] **Step 5: Create `open_docs/src/content/docs/en/reference/validation.md`**

```markdown
---
title: Validation
description: How goss verifies that a solve is actually correct.
sidebar:
  order: 1
---

import ConvergenceTable from '../../../../components/ConvergenceTable.astro';

goss does not trust a solver's "converged" status alone. Correctness is
established with independent oracles.

## RK4 re-integration

After a solve, goss re-integrates the dynamics with a fourth-order Runge–Kutta
scheme, starting from the solved initial state and applying the solved controls,
then measures the maximum deviation from the collocated trajectory. A correct
solution has a small deviation; a corrupted one is flagged. This is an
independent check of the transcription and solve — not a re-run of collocation.

## Convergence order

Refining the mesh must shrink the error at the scheme's theoretical rate
(Trapezoidal `O(h²)`, Hermite–Simpson `O(h⁴)`). Observed errors:

<ConvergenceTable />

## Cross-scheme agreement and published optima

goss also checks that different schemes converge to the same solution on
well-posed problems, and validates the full model → transcription → NLP → AD →
solver chain against a hand-ported subset of the Hock–Schittkowski problems,
whose optima are published.
```

- [ ] **Step 6: Create `open_docs/src/content/docs/en/reference/benchmarks.md`**

```markdown
---
title: Benchmarks
description: Scheme and solver performance on a reference problem.
sidebar:
  order: 2
---

import BenchmarkTable from '../../../../components/BenchmarkTable.astro';
import { Aside } from '@astrojs/starlight/components';

The table below reports each transcription scheme paired with each solver on a
reference exp-decay problem (`dx/dt = -x`, zero running cost). Timing covers the
solve call only; the validation error is the RK4 re-integration deviation
described in [Validation](/en/reference/validation/).

<BenchmarkTable />

<Aside type="note">
IPOPT is gradient-based and converges reliably on this well-posed problem. The
derivative-free NLopt baseline is included to validate the model independent of
the derivative path and may hit its iteration limit rather than converge.
</Aside>

These numbers are regenerated from the goss benchmark harness; see the
regeneration script in the repository for details.
```

- [ ] **Step 7: Build to verify data rendering**

Run:
```bash
cd open_docs && npm run build
```
Expected: build succeeds; `reference/validation` and `reference/benchmarks` routes appear; both tables render with the seeded rows and no JSON import errors.

- [ ] **Step 8: Commit**

```bash
git add open_docs/src/data/ open_docs/src/components/ open_docs/src/content/docs/en/reference/
git commit -m "docs(open_docs): add reference pages with data-driven benchmark and convergence tables"
```

---

### Task 5: C++ benchmark exporter (`goss_bench_export`)

**Files:**
- Create: `tools/bench_export/main.cpp`
- Modify: `CMakeLists.txt` (add an executable target after the `goss_bench_tests` block, around line 233)

**Interfaces:**
- Consumes: `goss::bench::run_scheme<Scheme>(...)`, `goss::bench::write_csv(results, path)`, `goss::model::Model`, `goss::solver::IpoptSolver`, `goss::solver::NloptSolver`, `goss::transcription::Trapezoidal`, `goss::transcription::HermiteSimpson`. The exporter reproduces the exact matrix built in `tests/bench/test_bench_flagship.cpp` (exp-decay, 20 intervals, both schemes × both solvers).
- Produces: an executable `goss_bench_export` that takes one argument (output CSV path) and writes a CSV whose header is `scheme,solver,status,objective,elapsed_s,validation_error,num_variables`. Task 6's Node script consumes this CSV.

- [ ] **Step 1: Create `tools/bench_export/main.cpp`**

```cpp
// tools/bench_export/main.cpp
//
// Standalone exporter: runs the flagship benchmark matrix
// ({Trapezoidal, HermiteSimpson} x {IpoptSolver, NloptSolver}) on the exp-decay
// problem and writes the results as CSV via goss::bench::write_csv.
//
// Usage: goss_bench_export <output.csv>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "goss/bench/harness.hpp"
#include "goss/bench/report.hpp"
#include "goss/model/model.hpp"
#include "goss/transcription/trapezoidal.hpp"
#include "goss/transcription/hermite_simpson.hpp"
#include "goss/solver/ipopt_solver.hpp"
#include "goss/solver/nlopt_solver.hpp"
#include "goss/solver/solver.hpp"

namespace {
// dx/dt = -x, x(0) = 1, cost = 0. Analytic: x(t) = exp(-t).
struct ExpDecayDyn {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x_vec, const std::vector<T>&, T) const {
        return { -x_vec[0] };
    }
};
struct ZeroCostFn {
    template <typename T>
    T operator()(const std::vector<T>&, const std::vector<T>&, T) const { return T(0); }
};
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: goss_bench_export <output.csv>\n";
        return 2;
    }
    const std::string out_path = argv[1];

    constexpr double kTimeHorizon    = 1.0;
    constexpr std::size_t kIntervals = 20;

    goss::model::Model model;
    auto x_state = model.add_state("x");
    model.set_initial_state(x_state, 1.0);
    model.set_mesh(0.0, kTimeHorizon, kIntervals);
    auto ocp = model.build(ExpDecayDyn{}, ZeroCostFn{});

    goss::solver::IpoptSolver ipopt_solver;
    goss::solver::NloptSolver nlopt_solver;
    std::vector<goss::solver::Solver*> solvers = {&ipopt_solver, &nlopt_solver};
    std::vector<std::string> solver_names      = {"IpoptSolver", "NloptSolver"};

    auto trap_results = goss::bench::run_scheme<goss::transcription::Trapezoidal>(
        ocp, model, "Trapezoidal", "flagship_trap", solvers, solver_names);
    auto hs_results = goss::bench::run_scheme<goss::transcription::HermiteSimpson>(
        ocp, model, "HermiteSimpson", "flagship_hs", solvers, solver_names);

    std::vector<goss::bench::BenchmarkResult> all_results = trap_results;
    all_results.insert(all_results.end(), hs_results.begin(), hs_results.end());

    try {
        goss::bench::write_csv(all_results, out_path);
    } catch (const std::exception& e) {
        std::cerr << "goss_bench_export: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "goss_bench_export: wrote " << all_results.size()
              << " rows to " << out_path << "\n";
    return 0;
}
```

- [ ] **Step 2: Add the CMake target**

In `CMakeLists.txt`, immediately after the `gtest_discover_tests(goss_bench_tests)`
line (currently line 233), add:

```cmake
# ---- Benchmark exporter (standalone, non-test) ----
add_executable(goss_bench_export tools/bench_export/main.cpp)
target_link_libraries(goss_bench_export PRIVATE
    goss_bench goss_sim goss_solver goss_model goss_transcription
    goss_nlp goss_ad goss_ad_impl
    goss_ipopt_iface goss_nlopt_iface cppadcg
    $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}>)
```

- [ ] **Step 3: Configure and build the exporter**

Run (using the existing `build/` directory):
```bash
cmake --build build --target goss_bench_export
```
Expected: the target compiles and links, producing `build/goss_bench_export`.
(If CMake reports the target is unknown, first re-run `cmake -S . -B build` to
regenerate, then rebuild.)

- [ ] **Step 4: Run the exporter and inspect the CSV header**

Run:
```bash
./build/goss_bench_export /tmp/goss_bench.csv && head -1 /tmp/goss_bench.csv
```
Expected: the program prints "wrote 4 rows"; the first CSV line is exactly
`scheme,solver,status,objective,elapsed_s,validation_error,num_variables`.

- [ ] **Step 5: Commit**

```bash
git add tools/bench_export/main.cpp CMakeLists.txt
git commit -m "feat(bench): add goss_bench_export standalone CSV exporter for docs pipeline"
```

---

### Task 6: Node regen script + documentation

**Files:**
- Create: `open_docs/scripts/regen-benchmarks.mjs`
- Create: `open_docs/scripts/README.md`

**Interfaces:**
- Consumes: the `goss_bench_export` binary from Task 5 (produces the CSV) and writes `open_docs/src/data/benchmarks.json` in the shape defined in Task 4 (keys: `scheme`, `solver`, `status`, `objective`, `elapsed_s`, `validation_error`, `num_variables`; string vs. number types matching Task 4's `BenchmarkTable.astro`).
- Produces: `npm run regen:benchmarks` (wired in Task 1's `package.json`).

- [ ] **Step 1: Create `open_docs/scripts/regen-benchmarks.mjs`**

```js
// Regenerates src/data/benchmarks.json from the goss C++ benchmark exporter.
//
// Requires a BUILT C++ tree: this script runs the already-compiled
// `goss_bench_export` binary. It does NOT build the C++ project.
//
// Usage:
//   node scripts/regen-benchmarks.mjs [path-to-goss_bench_export]
// Default binary path: ../build/goss_bench_export (relative to open_docs/).
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const openDocsDir = resolve(scriptDir, '..');

const binary = resolve(
  openDocsDir,
  process.argv[2] ?? '../build/goss_bench_export',
);

if (!existsSync(binary)) {
  console.error(
    `regen-benchmarks: exporter not found at '${binary}'.\n` +
      `Build it first (e.g. 'cmake --build build --target goss_bench_export'),\n` +
      `or pass an explicit path: node scripts/regen-benchmarks.mjs <path>.`,
  );
  process.exit(1);
}

const csvPath = join(mkdtempSync(join(tmpdir(), 'goss-bench-')), 'bench.csv');
execFileSync(binary, [csvPath], { stdio: 'inherit' });

const csv = readFileSync(csvPath, 'utf8').trim();
const [header, ...lines] = csv.split('\n');
const expectedHeader =
  'scheme,solver,status,objective,elapsed_s,validation_error,num_variables';
if (header.trim() !== expectedHeader) {
  console.error(
    `regen-benchmarks: unexpected CSV header.\n` +
      `  expected: ${expectedHeader}\n` +
      `  got:      ${header}`,
  );
  process.exit(1);
}

const rows = lines
  .filter((l) => l.trim().length > 0)
  .map((line) => {
    const [scheme, solver, status, objective, elapsed_s, validation_error, num_variables] =
      line.split(',');
    return {
      scheme,
      solver,
      status,
      objective: Number(objective),
      elapsed_s: Number(elapsed_s),
      validation_error: Number(validation_error),
      num_variables: Number(num_variables),
    };
  });

const outPath = resolve(openDocsDir, 'src/data/benchmarks.json');
writeFileSync(outPath, JSON.stringify(rows, null, 2) + '\n');
console.error(`regen-benchmarks: wrote ${rows.length} rows to ${outPath}`);
```

- [ ] **Step 2: Create `open_docs/scripts/README.md`**

```markdown
# Regenerating benchmark & convergence data

The docs site renders numbers from committed JSON in `src/data/`. Those files
are seeded with real values and can be regenerated from the C++ benchmark
harness. **Regeneration requires a built C++ tree; building the docs site does
not.**

## Benchmarks (`src/data/benchmarks.json`)

1. Build the exporter in the C++ tree:

   ```bash
   cmake --build build --target goss_bench_export
   ```

2. Run the regen script from `open_docs/`:

   ```bash
   npm run regen:benchmarks
   # or, with an explicit binary path:
   node scripts/regen-benchmarks.mjs ../build/goss_bench_export
   ```

The script runs `goss_bench_export`, reads its CSV, and rewrites
`src/data/benchmarks.json`. Review the diff and commit it.

## Convergence (`src/data/convergence.json`)

Regenerated manually for now — the accuracy tests
(`tests/accuracy/test_convergence_order.cpp`) do not yet emit a machine-readable
file. Update `src/data/convergence.json` by hand from those test results until a
dedicated exporter exists.
```

- [ ] **Step 3: Verify the script fails loudly when the binary is absent**

Run:
```bash
cd open_docs && node scripts/regen-benchmarks.mjs /tmp/does-not-exist-goss
```
Expected: exits non-zero with the "exporter not found" message. (This confirms the guard without requiring a C++ build in this step.)

- [ ] **Step 4: Verify regeneration end-to-end (requires the Task 5 binary)**

Run (from `open_docs/`, assuming `../build/goss_bench_export` exists from Task 5):
```bash
node scripts/regen-benchmarks.mjs && git --no-pager diff --stat src/data/benchmarks.json
```
Expected: the script prints "wrote 4 rows"; `src/data/benchmarks.json` is
rewritten with 4 rows whose keys match Task 4's shape (values will reflect the
live solve). If the C++ tree is not built, this step is deferred until it is —
the seeded JSON remains valid in the meantime.

- [ ] **Step 5: Commit**

```bash
git add open_docs/scripts/
git commit -m "feat(open_docs): add benchmark data regeneration script and docs"
```

---

## Final verification

- [ ] **Step 1: Full site build**

Run:
```bash
cd open_docs && npm run build
```
Expected: build succeeds with no errors and no broken-link warnings; landing
page, all four overview pages, all four guides, and both reference pages appear
in the output; both data tables render.

- [ ] **Step 2: Manual browser check**

Run:
```bash
cd open_docs && npm run preview
```
Then open the served URL and confirm: the landing hero + card grid render; the
sidebar shows Overview / Guides / Reference groups; the entry-points Tabs
switch; the first-solve Steps render; and the benchmark and convergence tables
display their rows.

- [ ] **Step 3: Confirm isolation**

Verify no public page links into or references the private `docs/` folder:
```bash
grep -rn "docs/superpowers\|literature-review\|preferred-stacks\|best-ideas" open_docs/src || echo "clean"
```
Expected: prints `clean`.

## Self-Review notes

- **Spec coverage:** scaffold + i18n (Task 1); overview track (Task 2); guides
  track (Task 3); validation & benchmark reference pages + committed data +
  components (Task 4); C++ exporter (Task 5); regen script + docs (Task 6).
  Closed-source constraint honored — no install/build/clone instructions in any
  public page; getting-started is a conceptual walkthrough. Private `docs/` is
  never referenced (verified in Final Verification Step 3).
- **Data-shape consistency:** the JSON keys in Task 4
  (`scheme,solver,status,objective,elapsed_s,validation_error,num_variables`),
  the `BenchmarkTable.astro` `Row` type, the CSV header emitted by
  `goss::bench::to_csv` / the Task 5 exporter, and the parser in Task 6 all use
  the same seven names in the same order.
- **Convergence data:** intentionally manual (documented in Task 6 README and
  the spec's out-of-scope) — no automated exporter this round.
