import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    lang: z.enum(['ko', 'en']),
    key: z.string(),
    tags: z.array(z.string()).default([]),
    tistory: z.string().url().optional(),
    draft: z.boolean().default(false),
    kind: z.enum(['report', 'review']).default('report'),
  }),
});

export const collections = { posts };
