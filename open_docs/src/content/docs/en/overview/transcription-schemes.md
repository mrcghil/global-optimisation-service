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
