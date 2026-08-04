// dashboard/src/lib/types.ts
// TypeScript mirror of the goss dashboard JSON contract emitted by
// export_dashboard_data (see include/goss/sim/dashboard_export.hpp). Kept in one
// place so every view and the dataClient share one definition of the shapes.

export interface Axis {
  parameter: string;
  values: number[];
}

/** Per-run summary as it appears in index.json / sweep manifests. */
export interface RunRef {
  run_id: string;
  problem: string;
  version: string;
  parameters: Record<string, number>;
  status: string;
  objective: number;
  label?: string;
}

export interface SweepSummary {
  slug: string;
  label: string;
  problem: string;
  version: string;
  axes: Axis[];
  num_runs: number;
  num_succeeded: number;
}

export interface CampaignSummary {
  name: string;
  slug: string;
  num_sweeps: number;
  num_runs: number;
  num_succeeded: number;
}

/** index.json */
export interface DashboardIndex {
  campaigns: CampaignSummary[];
  sweeps: SweepSummary[];
}

/** sweep/<slug>.json */
export interface SweepDetail {
  label: string;
  combinator: string;
  problem: string;
  version: string;
  axes: Axis[];
  num_runs: number;
  num_succeeded: number;
  runs: RunRef[];
}

export interface Trajectory {
  time: number[];
  states: Record<string, number[]>;
  controls: Record<string, number[]>;
}

/** run/<run_id>.json */
export interface RunDetail {
  run_id: string;
  problem: string;
  version: string;
  spec: unknown;
  result: { status: string; objective: number; message: string };
  diagnosis: { ok: boolean; summary: string; advice: string };
  provenance: {
    created_utc: string;
    hostname: string;
    scheme: string;
    goss_version: string;
  };
  trajectory: Trajectory;
}
