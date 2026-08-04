// dashboard/src/lib/dataClient.ts
// The single seam between the UI and the data source.  Every view fetches
// through here, so moving from the local static export to a remote server is a
// one-line config change (PUBLIC_GOSS_DATA_BASE), never a component edit.
//
//   local (default):  files under /data  (Astro serves public/data)
//   remote (later):   set PUBLIC_GOSS_DATA_BASE=https://host/api at build/run
//
// Both sources MUST return the identical contract shapes (see types.ts); the C++
// exporter and any future server are held to that same contract.
import type { DashboardIndex, RunDetail, SweepDetail } from './types';

const RAW_BASE =
  (import.meta.env.PUBLIC_GOSS_DATA_BASE as string | undefined) ?? '/data';

/** Base URL with any trailing slash removed. */
const BASE = RAW_BASE.replace(/\/+$/, '');

async function getJson<T>(path: string): Promise<T> {
  const url = `${BASE}/${path}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`dataClient: ${url} -> HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export const dataClient = {
  base: BASE,
  index: () => getJson<DashboardIndex>('index.json'),
  sweep: (slug: string) => getJson<SweepDetail>(`sweep/${slug}.json`),
  run: (runId: string) => getJson<RunDetail>(`run/${runId}.json`),
};
