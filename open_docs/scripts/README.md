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

Regenerated manually for now — the convergence-order accuracy tests do not yet
emit a machine-readable file. Update `src/data/convergence.json` by hand from
those test results until a dedicated exporter exists.
