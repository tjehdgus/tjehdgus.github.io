import { getCollection } from 'astro:content';
import { projects, type Lang } from './site';

const order = projects.map((p) => p.postKey);

export async function postsFor(lang: Lang) {
  const all = await getCollection('posts', (p) => p.data.lang === lang && !p.data.draft);
  const rank = (k: string) => {
    const i = order.indexOf(k);
    return i === -1 ? 999 : i;
  };
  return all.sort((a, b) => b.data.date.getTime() - a.data.date.getTime() || rank(a.data.key) - rank(b.data.key));
}
