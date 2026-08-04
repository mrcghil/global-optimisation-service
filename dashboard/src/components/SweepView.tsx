// dashboard/src/components/SweepView.tsx
// Sweep view: the sweep's OWN named axes drive the UI (no hardcoded params).
// - 2 axes  -> objective heatmap (x = axis0, y = axis1), cells link to the run
// - else    -> a sortable table of runs
// Selecting a cell/row navigates to the run drill-down.
import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { dataClient } from '../lib/dataClient';
import type { RunRef, SweepDetail } from '../lib/types';

const MARGIN = { top: 30, right: 90, bottom: 56, left: 72 };
const CELL = 68;

function runHref(runId: string): string {
  return `${import.meta.env.BASE_URL}run/?id=${runId}`.replace('//run', '/run');
}

function Heatmap({ detail }: { detail: SweepDetail }) {
  const ref = useRef<SVGSVGElement | null>(null);
  const [ax, ay] = detail.axes;

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const xs = ax.values.map(String);
    const ys = ay.values.map(String);
    const innerW = xs.length * CELL;
    const innerH = ys.length * CELL;

    const x = d3.scaleBand<string>().domain(xs).range([0, innerW]).padding(0.04);
    const y = d3.scaleBand<string>().domain(ys).range([0, innerH]).padding(0.04);

    // Look up a run by its (axis0, axis1) parameter values.
    const byKey = new Map<string, RunRef>();
    for (const run of detail.runs) {
      const key = `${run.parameters[ax.parameter]}|${run.parameters[ay.parameter]}`;
      byKey.set(key, run);
    }

    const objectives = detail.runs
      .filter((r) => r.status === 'Success')
      .map((r) => r.objective);
    const color = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([d3.min(objectives) ?? 0, d3.max(objectives) ?? 1]);

    const g = svg
      .attr('viewBox', `0 0 ${innerW + MARGIN.left + MARGIN.right} ${innerH + MARGIN.top + MARGIN.bottom}`)
      .attr('width', '100%')
      .style('height', 'auto')
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x));
    g.append('g').call(d3.axisLeft(y));
    g.append('text')
      .attr('x', innerW / 2).attr('y', innerH + 44)
      .attr('text-anchor', 'middle').attr('class', 'axis-label')
      .text(ax.parameter);
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2).attr('y', -52)
      .attr('text-anchor', 'middle').attr('class', 'axis-label')
      .text(ay.parameter);

    for (const xv of ax.values) {
      for (const yv of ay.values) {
        const run = byKey.get(`${xv}|${yv}`);
        const cell = g
          .append('a')
          .attr('href', run ? runHref(run.run_id) : null)
          .append('rect')
          .attr('x', x(String(xv))!)
          .attr('y', y(String(yv))!)
          .attr('width', x.bandwidth())
          .attr('height', y.bandwidth())
          .attr('rx', 4)
          .attr('fill',
            run && run.status === 'Success'
              ? (color(run.objective) as string)
              : '#3a3a3a')
          .attr('class', 'cell');
        cell.append('title').text(
          run
            ? `${ax.parameter}=${xv}, ${ay.parameter}=${yv}\nobjective=${run.objective.toFixed(4)}\nstatus=${run.status}`
            : `${ax.parameter}=${xv}, ${ay.parameter}=${yv}\n(no run)`,
        );
      }
    }
  }, [detail, ax, ay]);

  return (
    <figure className="chart">
      <figcaption>Objective over {ax.parameter} × {ay.parameter} (click a cell)</figcaption>
      <svg ref={ref} role="img" aria-label="sweep objective heatmap" />
    </figure>
  );
}

function RunTable({ detail }: { detail: SweepDetail }) {
  const params = detail.axes.map((a) => a.parameter);
  return (
    <table className="runs">
      <thead>
        <tr>
          {params.map((p) => <th key={p}>{p}</th>)}
          <th>status</th>
          <th>objective</th>
        </tr>
      </thead>
      <tbody>
        {detail.runs.map((run) => (
          <tr key={run.run_id}>
            {params.map((p) => <td key={p}>{run.parameters[p]}</td>)}
            <td className={run.status === 'Success' ? 'ok' : 'bad'}>{run.status}</td>
            <td>
              <a href={runHref(run.run_id)}>{run.objective.toFixed(4)}</a>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function SweepView({ slug }: { slug: string }) {
  const [detail, setDetail] = useState<SweepDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    dataClient.sweep(slug).then(setDetail).catch((e) => setError(String(e)));
  }, [slug]);

  const twoAxis = useMemo(() => detail?.axes.length === 2, [detail]);

  if (error) return <p className="error">{error}</p>;
  if (!detail) return <p>Loading sweep…</p>;

  return (
    <section>
      <p className="meta">
        {detail.problem}@{detail.version} · {detail.combinator} ·{' '}
        {detail.num_succeeded}/{detail.num_runs} converged
      </p>
      {twoAxis ? <Heatmap detail={detail} /> : <RunTable detail={detail} />}
    </section>
  );
}
