// dashboard/src/components/RunClient.tsx
// Reads ?id= from the URL at runtime and renders the run drill-down.
import { useEffect, useState } from 'react';
import RunView from './RunView';

export default function RunClient() {
  const [id, setId] = useState<string | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setId(params.get('id'));
  }, []);

  if (id === null) return <p>Loading…</p>;
  if (!id) return <p className="error">Missing ?id= parameter.</p>;
  return <RunView runId={id} />;
}
