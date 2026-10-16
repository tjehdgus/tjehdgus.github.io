---
title: "Product inquiry RAG chatbot: tech stack and components"
description: "The technical makeup of a customer product-inquiry RAG chatbot built at a previous company. Company data and internal figures are not covered."
date: 2026-04-14
lang: en
key: product-rag-chatbot
tags: [RAG, ChromaDB, bge-m3, PII Masking, FastAPI, WebSocket]
tistory: https://mynews76154.tistory.com/13
---

## Overview

At a previous company I built a RAG chatbot that answers customer product questions from internal documents and hands requests that need a person over to a consultation form. This post only lists which technologies were used and where. Data, evaluation results, prompt contents and internal system details are out of scope.

## Tech stack

| Area | Technology |
|---|---|
| Collection and parsing | Selenium, BeautifulSoup, python-docx |
| Embeddings | BAAI/bge-m3 (compared: multilingual-e5-large, ko-sroberta-multitask) |
| Vector DB | ChromaDB |
| Orchestration | LangChain |
| LLM | Azure OpenAI (behind a swappable LLM interface) |
| Privacy | Microsoft Presidio + Korean pattern recognizers |
| Backend | Python, FastAPI, WebSocket |
| Sessions | Redis / DB / in-memory fallback |
| Evaluation | Hit@k, MRR@k, RAGAS |

## Architecture

```text
[Indexing]
Web crawl / DOCX / JSON
  → parse into a common section schema
  → chunking → metadata / context enrichment
  → embeddings → per-source vector collections

[Answering]
question → input checks → session history
  ├─ vector search → send sources
  └─ PII masking → prompt → LLM streaming → PII restore → send

[Follow-up]
consultation request → lead stored → admin summary and classification
```

## Components

- **Normalisation.** Sources in different formats (web tables, DOCX, JSON) were mapped to one section schema. Merged table cells were expanded into a grid to build path-style column names.
- **Per-source collections.** Each source has its own vector collection so it can be re-indexed alone; results are merged by distance.
- **Chunking and enrichment.** Section-level and recursive splitting were compared, along with prepending metadata tags or LLM-written context sentences (Contextual Retrieval). Context generation is hash-cached to avoid repeat calls.
- **Retrieval experiment pipeline.** Experiment code that sweeps embedding model × chunking × enrichment combinations and compares Hit@k and MRR@k.
- **PII masking.** Presidio replaces personal data with placeholders before the LLM call; the streamed answer is buffered so tags are not split, then restored.
- **Multi-turn.** A session manager that falls back from Redis to DB to memory feeds recent turns into the prompt.
- **Streaming.** FastAPI WebSocket sends source, token and end events in order, with input length limits and rate limiting.
- **Handoff.** Unsupported or human-only requests are stored as consultation leads, and an admin view uses the LLM to summarise and classify the conversation.

## References

- Build log (Korean): https://mynews76154.tistory.com/13
- BAAI/bge-m3, intfloat/multilingual-e5-large, jhgan/ko-sroberta-multitask
- ChromaDB, LangChain, FastAPI, Redis, Microsoft Presidio, RAGAS
- Anthropic, "Contextual Retrieval"
