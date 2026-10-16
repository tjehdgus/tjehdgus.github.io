import { getCollection } from 'astro:content';
import type { Lang } from './site';

export async function postsFor(lang: Lang) {
  const all = await getCollection('posts', (p) => p.data.lang === lang && !p.data.draft);
  return all.sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}
