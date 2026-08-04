// dashboard/src/components/IndexView.tsx
// Campaign/sweep index: the entry point. Lists campaigns and standalone sweeps
// with success counts; each links to its sweep parameter map.
import { useEffect, useState } from 'react';
import { dataClient } from '../lib/dataClient';
import type { DashboardIndex } from '../lib/types';

function sweepHref(slug: string): string {
  return `${import.meta.env.BASE_URL}sweep/?slug=${slug}`.replace('//sweep', '/sweep');
}

export default function IndexView() {
  const [index, setIndex] = useState<DashboardIndex | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    dataClient.index().then(setIndex).catch((e) => setError(String(e)));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!index) return <p>Loading…</p>;

  return (
    <div>
      <p className="meta">Data source: <code>{dataClient.base}</code></p>

      <h2>Campaigns</h2>
      {index.campaigns.length === 0 && <p className="muted">No campaigns.</p>}
      <ul className="cards">
        {index.campaigns.map((c) => (
          <li key={c.slug} className="card">
            <strong>{c.name}</strong>
            <span className="stat">
              {c.num_sweeps} sweeps · {c.num_succeeded}/{c.num_runs} runs converged
            </span>
          </li>
        ))}
      </ul>

      <h2>Sweeps</h2>
      {index.sweeps.length === 0 && <p className="muted">No sweeps.</p>}
      <ul className="cards">
        {index.sweeps.map((s) => (
          <li key={s.slug} className="card">
            <a href={sweepHref(s.slug)}><strong>{s.label}</strong></a>
            <span className="stat">
              {s.problem}@{s.version} ·{' '}
              {s.axes.map((a) => a.parameter).join(' × ')} ·{' '}
              {s.num_succeeded}/{s.num_runs} converged
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
