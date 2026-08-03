// Regenerates src/data/benchmarks.json from the goss C++ benchmark exporter.
//
// Requires a BUILT C++ tree: this script runs the already-compiled
// `goss_bench_export` binary. It does NOT build the C++ project.
//
// Usage:
//   node scripts/regen-benchmarks.mjs [path-to-goss_bench_export]
// Default binary path: ../build/goss_bench_export (relative to open_docs/).
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const openDocsDir = resolve(scriptDir, '..');

const binary = resolve(
  openDocsDir,
  process.argv[2] ?? '../build/goss_bench_export',
);

if (!existsSync(binary)) {
  console.error(
    `regen-benchmarks: exporter not found at '${binary}'.\n` +
      `Build it first (e.g. 'cmake --build build --target goss_bench_export'),\n` +
      `or pass an explicit path: node scripts/regen-benchmarks.mjs <path>.`,
  );
  process.exit(1);
}

const csvPath = join(mkdtempSync(join(tmpdir(), 'goss-bench-')), 'bench.csv');
execFileSync(binary, [csvPath], { stdio: 'inherit' });

const csv = readFileSync(csvPath, 'utf8').trim();
const [header, ...lines] = csv.split('\n');
const expectedHeader =
  'scheme,solver,status,objective,elapsed_s,validation_error,num_variables';
if (header.trim() !== expectedHeader) {
  console.error(
    `regen-benchmarks: unexpected CSV header.\n` +
      `  expected: ${expectedHeader}\n` +
      `  got:      ${header}`,
  );
  process.exit(1);
}

const rows = lines
  .filter((l) => l.trim().length > 0)
  .map((line) => {
    const [scheme, solver, status, objective, elapsed_s, validation_error, num_variables] =
      line.split(',');
    return {
      scheme,
      solver,
      status,
      objective: Number(objective),
      elapsed_s: Number(elapsed_s),
      validation_error: Number(validation_error),
      num_variables: Number(num_variables),
    };
  });

const outPath = resolve(openDocsDir, 'src/data/benchmarks.json');
writeFileSync(outPath, JSON.stringify(rows, null, 2) + '\n');
console.error(`regen-benchmarks: wrote ${rows.length} rows to ${outPath}`);
