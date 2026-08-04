// dashboard/src/components/SweepClient.tsx
// Reads ?slug= from the URL at runtime and renders the sweep view. Kept separate
// from SweepView so the query-param plumbing (client-only) is isolated and the
// view itself stays a pure slug -> UI component.
import { useEffect, useState } from 'react';
import SweepView from './SweepView';

export default function SweepClient() {
  const [slug, setSlug] = useState<string | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setSlug(params.get('slug'));
  }, []);

  if (slug === null) return <p>Loading…</p>;
  if (!slug) return <p className="error">Missing ?slug= parameter.</p>;
  return <SweepView slug={slug} />;
}
