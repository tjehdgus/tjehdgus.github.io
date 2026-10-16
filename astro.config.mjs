import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://tjehdgus.github.io',
  integrations: [sitemap({ i18n: { defaultLocale: 'ko', locales: { ko: 'ko-KR', en: 'en-US' } } })],
  i18n: {
    defaultLocale: 'ko',
    locales: ['ko', 'en'],
    routing: { prefixDefaultLocale: false },
  },
  markdown: {
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark-dimmed' } },
  },
});
