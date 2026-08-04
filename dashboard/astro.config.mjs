// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

// A plain SPA-style static site: three views over the exported JSON contract.
// The React islands do all data fetching at runtime via src/lib/dataClient, so
// switching from the local static /data to a remote API is a config change
// (PUBLIC_GOSS_DATA_BASE), never a rebuild of the components.
export default defineConfig({
  integrations: [react()],
  server: { port: 4330 },
});
