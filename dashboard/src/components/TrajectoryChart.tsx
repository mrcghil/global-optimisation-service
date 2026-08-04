// dashboard/src/components/TrajectoryChart.tsx
// A D3 multi-series line chart for one trajectory group (states OR controls).
// D3 owns the SVG subtree via a ref; React only decides when to (re)render.
import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface Props {
  time: number[];
  series: Record<string, number[]>;
  title: string;
  yLabel?: string;
}

const MARGIN = { top: 24, right: 120, bottom: 40, left: 56 };
const WIDTH = 680;
const HEIGHT = 300;

export default function TrajectoryChart({ time, series, title, yLabel }: Props) {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const names = Object.keys(series);
    if (!ref.current || time.length === 0 || names.length === 0) return;

    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const innerW = WIDTH - MARGIN.left - MARGIN.right;
    const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;

    const x = d3
      .scaleLinear()
      .domain(d3.extent(time) as [number, number])
      .range([0, innerW]);

    const allValues = names.flatMap((n) => series[n]);
    const y = d3
      .scaleLinear()
      .domain([d3.min(allValues) ?? 0, d3.max(allValues) ?? 1])
      .nice()
      .range([innerH, 0]);

    const color = d3.scaleOrdinal(d3.schemeTableau10).domain(names);

    const g = svg
      .attr('viewBox', `0 0 ${WIDTH} ${HEIGHT}`)
      .attr('width', '100%')
      .style('height', 'auto')
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6));
    g.append('g').call(d3.axisLeft(y).ticks(6));

    // Axis labels.
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 34)
      .attr('text-anchor', 'middle')
      .attr('class', 'axis-label')
      .text('time');
    if (yLabel) {
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('class', 'axis-label')
        .text(yLabel);
    }

    const line = d3
      .line<number>()
      .x((_, i) => x(time[i]))
      .y((d) => y(d));

    for (const name of names) {
      g.append('path')
        .datum(series[name])
        .attr('fill', 'none')
        .attr('stroke', color(name) as string)
        .attr('stroke-width', 2)
        .attr('d', line);
    }

    // Legend.
    const legend = g
      .append('g')
      .attr('transform', `translate(${innerW + 16},0)`);
    names.forEach((name, i) => {
      const row = legend.append('g').attr('transform', `translate(0,${i * 20})`);
      row
        .append('rect')
        .attr('width', 12)
        .attr('height', 12)
        .attr('fill', color(name) as string);
      row
        .append('text')
        .attr('x', 18)
        .attr('y', 10)
        .attr('class', 'legend-label')
        .text(name);
    });
  }, [time, series, yLabel]);

  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg ref={ref} role="img" aria-label={title} />
    </figure>
  );
}
