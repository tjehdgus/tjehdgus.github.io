import rss from '@astrojs/rss';
import { postsFor } from '../../data/posts';
export async function GET(context) {
  const posts = await postsFor('en');
  return rss({
    title: 'Donghyun Seo · AI Engineer',
    description: 'RAG, agents and ML project write-ups',
    site: context.site,
    items: posts.map((p) => ({ title: p.data.title, description: p.data.description, pubDate: p.data.date, link: '/en/posts/' + p.data.key + '/' })),
  });
}
