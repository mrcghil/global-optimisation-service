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
