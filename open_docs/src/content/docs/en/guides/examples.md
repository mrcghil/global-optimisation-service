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
