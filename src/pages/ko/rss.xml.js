import rss from '@astrojs/rss';
import { postsFor } from '../../data/posts';
export async function GET(context) {
  const posts = await postsFor('ko');
  return rss({
    title: '서동현 · AI 엔지니어',
    description: 'RAG·에이전트·ML 프로젝트 정리',
    site: context.site,
    items: posts.map((p) => ({ title: p.data.title, description: p.data.description, pubDate: p.data.date, link: '/ko/posts/' + p.data.key + '/' })),
  });
}
