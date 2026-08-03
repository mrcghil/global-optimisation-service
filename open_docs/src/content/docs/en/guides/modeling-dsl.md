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
