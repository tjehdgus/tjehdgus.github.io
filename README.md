# tjehdgus.github.io

서동현의 한·영 포트폴리오 & 대표 프로젝트 글. Astro로 만들고 GitHub Actions로 GitHub Pages에 배포한다.

- English: `/` (기본) · 한국어: `/ko/` · 옛 `/en/*` 주소는 `/`로 리다이렉트
- 상세한 개발 기록과 회고: https://mynews76154.tistory.com

## 구조

```
src/
  content/posts/ko/*.md   한국어 글 (원본)
  content/posts/en/*.md   영어판 (같은 key)
  data/site.ts            UI 문구, 프로젝트 목록 (ko/en)
  views/                  페이지 공통 화면
  pages/, pages/ko/       언어별 라우트 (pages/en/은 리다이렉트용)
```

## 글 추가 규칙

1. 한국어판을 원본으로 `src/content/posts/ko/<key>.md`에 쓴다.
2. 같은 `key`로 영어판을 `src/content/posts/en/<key>.md`에 만든다.
3. 한국어판을 고치면 영어판도 같은 커밋에서 고치고, 두 파일의 `updated`를 같이 바꾼다.

frontmatter:

```yaml
title: "..."
description: "..."
date: 2026-10-16
updated: 2026-10-20   # 선택
lang: ko              # ko | en
key: kftc-rag         # 두 언어판을 잇는 키 = URL
tags: [RAG, LoRA]
tistory: https://mynews76154.tistory.com/1   # 선택, 개발 과정 원문
```

## 로컬 실행

```bash
npm install
npm run dev
```

`main`에 push하면 `.github/workflows/deploy.yml`이 빌드해서 배포한다.
