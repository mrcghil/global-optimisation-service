// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  integrations: [
    starlight({
      title: 'goss',
      description:
        'A C++ direct-collocation solver for trajectory optimization and nonlinear optimal control.',
      defaultLocale: 'en',
      locales: {
        en: { label: 'English' },
      },
      sidebar: [
        { label: 'Overview', autogenerate: { directory: 'overview' } },
        { label: 'Guides', autogenerate: { directory: 'guides' } },
        { label: 'Reference', autogenerate: { directory: 'reference' } },
      ],
    }),
  ],
});
