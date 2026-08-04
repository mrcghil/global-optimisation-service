// dashboard/src/components/RunView.tsx
// Run drill-down: trajectories (states + controls), the convergence diagnosis,
// parameters, and provenance for a single run.
import { useEffect, useState } from 'react';
import { dataClient } from '../lib/dataClient';
import type { RunDetail } from '../lib/types';
import TrajectoryChart from './TrajectoryChart';

export default function RunView({ runId }: { runId: string }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    dataClient.run(runId).then(setRun).catch((e) => setError(String(e)));
  }, [runId]);

  if (error) return <p className="error">{error}</p>;
  if (!run) return <p>Loading run…</p>;

  const params = (run.spec as { parameters?: Record<string, number> })?.parameters ?? {};

  return (
    <section>
      <p className="meta">
        {run.problem}@{run.version} · <code>{run.run_id}</code> ·{' '}
        scheme {run.provenance.scheme}
      </p>

      <div className={`banner ${run.diagnosis.ok ? 'ok' : 'bad'}`}>
        <strong>{run.result.status}</strong> — objective{' '}
        {run.result.objective.toFixed(6)}. {run.diagnosis.summary}
        {run.diagnosis.advice && <div className="advice">{run.diagnosis.advice}</div>}
      </div>

      <div className="params">
        {Object.entries(params).map(([k, v]) => (
          <span key={k} className="pill">{k} = {String(v)}</span>
        ))}
      </div>

      <TrajectoryChart
        time={run.trajectory.time}
        series={run.trajectory.states}
        title="States"
        yLabel="state value"
      />
      <TrajectoryChart
        time={run.trajectory.time}
        series={run.trajectory.controls}
        title="Controls"
        yLabel="control value"
      />

      <details className="provenance">
        <summary>Provenance</summary>
        <dl>
          <dt>created</dt><dd>{run.provenance.created_utc || '—'}</dd>
          <dt>host</dt><dd>{run.provenance.hostname || '—'}</dd>
          <dt>scheme</dt><dd>{run.provenance.scheme}</dd>
          <dt>goss</dt><dd>{run.provenance.goss_version || '—'}</dd>
        </dl>
      </details>
    </section>
  );
}
