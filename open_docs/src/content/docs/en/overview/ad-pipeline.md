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
