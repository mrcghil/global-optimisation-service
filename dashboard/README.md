# goss results dashboard

A static Astro app that visualizes goss solve results — single runs, sweeps, and
campaigns — from the JSON data contract emitted by `goss_dashboard_export`.

## Three views

- **Index** (`/`) — campaigns and standalone sweeps with converged counts.
- **Sweep** (`/sweep/?slug=<slug>`) — the sweep's own named axes drive the UI:
  a 2-axis sweep renders a D3 objective heatmap (click a cell → run); other
  shapes render a run table.
- **Run** (`/run/?id=<run_id>`) — state/control trajectories (D3), the
  convergence diagnosis, parameters, and provenance.

## Data flow

```
solve → write_run/write_sweep/write_campaign   (goss::sim, HDF5 + JSON manifests)
      → goss_dashboard_export <results> <out>   (emits the JSON contract)
      → dashboard/public/data/                  (index.json, sweep/*, run/*)
      → Astro app (D3 views via src/lib/dataClient)
```

Every view fetches through `src/lib/dataClient.ts`, the single seam to the data
source. It reads from `PUBLIC_GOSS_DATA_BASE` (default `/data`, i.e. the static
files under `public/data`).

## Local → remote

The move to remote runs is a **config change, not a rewrite**: stand up a server
that serves the *same* contract shapes and point the app at it —

```bash
PUBLIC_GOSS_DATA_BASE=https://host/api npm run build
```

No component changes: the C++ exporter and any future server are both held to the
contract in `src/lib/types.ts`.

## Develop

```bash
npm install
npm run dev        # predev copies sample-data/data → public/data
```

Regenerate the sample fixtures from C++ (needs an HDF5-enabled build):

```bash
# from repo root, inside the build container
./build/goss_dashboard_export <results_root> dashboard/sample-data/data
```
