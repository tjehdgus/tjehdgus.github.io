---
title: "Product-Inquiry RAG Chatbot for a Manufacturer: Per-Source Indexing, PII-Masked Streaming, and a Chunking/Enrichment Sweep"
description: "A product-inquiry chatbot that indexes a DOCX catalog, JSON specs, and crawled website data into per-source ChromaDB collections, streams LLM answers over WebSocket from PII-masked questions and restores the PII in the stream, plus the design of its retrieval experiments."
date: 2026-10-16
lang: en
key: product-rag-chatbot
tags: [RAG, ChromaDB, bge-m3, PII Masking, FastAPI, Contextual Retrieval]
tistory: https://mynews76154.tistory.com/13
---

## Abstract

At my previous company, a manufacturer, I built a RAG chatbot that handles first-line customer product inquiries. A product catalog (DOCX), numeric product specs (JSON), and crawled website data were parsed into sections, embedded with bge-m3, and indexed in ChromaDB. A FastAPI WebSocket server streams the answers; personal information in the question is replaced with placeholders via Presidio before it reaches the LLM and is restored in the output stream. Conversation history is held by a session manager that falls back from Redis to a database to process memory, and requests without supporting documents, or for quotes and technical documents, are routed to a consultation form and stored as leads. To choose retrieval settings, I built an experiment pipeline that sweeps 3 embedding models × 122 chunking strategies × 3 enrichment modes, scored by Hit@5 and MRR@5. As a reported figure, first-response time for inquiries dropped from about 15 minutes to under 10 seconds.

Company names, product names, data samples, and paths are generalized, and file names and identifying details in the quoted code are removed.

## 1. Problem

**Task.** When a customer asks in web chat about product usage, suitable products, property values, or office and dealer contacts, answer from internal documents. Requests that a staff member must handle (pricing, safety and technical documents, unrelated inquiries) are routed to a consultation request instead.

| Item | Description |
|---|---|
| Input | Customer question (Korean/English, up to 2,000 chars), prior turns in the session |
| Output | Streamed answer, list of source documents, consultation form when needed |
| Side outputs | Consultation leads, conversation summary and inquiry type (5 classes) |

**Metrics.** Each test question is labeled with a target product name and scored on the top-k retrieved documents. The four RAGAS metrics (context precision/recall, faithfulness, answer relevancy) can optionally be run for generation quality.

```text
hit(q) = 1  if any of the top-k documents contains the target product name, else 0
Hit@k  = sum(hit(q)) / N
MRR@k  = sum(1 / rank(q)) / N     (0 when there is no hit)
k = 5
```

**Constraints.** Raw customer PII must not be sent to the external LLM API. Information not in the documents (including made-up URLs, emails, phone numbers) must not be generated. Sources are updated on different schedules, so a change in one source should only require re-indexing that source.

## 2. Data

All three sources were normalized into a shared dataclass `ParsedSection(source, category, product_name, sub_section, content, section_type, extra_metadata)`. The figures below come from running the loaders on the catalog and spec files committed to the repository. The company-info JSON is excluded from the repository, so its figures could not be verified.

| Source | Format | Parsing unit | Mean length (chars) | Range (chars) |
|---|---|---|---|---|
| Product catalog | Narrative DOCX | Heading 1/2/3 | 329 | 85 – 981 |
| Numeric specs | JSON (crawled tables) | One product | 291 | 162 – 388 |
| Company / offices / dealers | JSON (crawled) | Greeting, org chart, history period, offices, dealer region | n/a | n/a |

Catalog sections consist of product descriptions, troubleshooting entries, FAQs and a category overview, most of them product descriptions. Section counts are omitted because they reveal the company's scale.

- **Crawling.** Tables were read with Selenium and BeautifulSoup. Multi-row headers merged with `rowspan`/`colspan` were expanded into a grid to build path-style column names such as `"Properties > pH"`.
- **Spec serialization.** Each product became a `[category] product - spec` header followed by `• attribute: value` lines. Path-style keys kept only their last segment; empty values and `-` were dropped.
- **DOCX parsing.** Paragraph styles were tracked to cut sections at Heading 2 (product). Heading 3 titles separated troubleshooting and FAQ sections, and separator-only paragraphs were skipped.

## 3. Method

### 3.1 Pipeline

```text
[Indexing]
Website crawl / catalog DOCX / spec JSON
  → section parsing (ParsedSection)
  → chunking (by_product | recursive_N)
  → enrichment (none | enriched | contextual)
  → bge-m3 embeddings → ChromaDB (collections: docx / product_spec / company)

[Answering, one WebSocket turn]
question → length and rate-limit checks → append to session history
  ├─ retrieval (top-k) → send source list
  └─ PII masking → prompt (rules + history + documents + question)
     → LLM streaming → PII restoration (buffered) → send tokens → save to history and DB

[Follow-up]
consultation request → create lead + link conversation → admin view summary and type
```

- Section parsing: maps heterogeneous sources onto one schema.
- Chunking and enrichment: splits sections and prepends metadata or a context sentence.
- Answering: places retrieved documents and recent turns into a rule-based prompt and streams the output.
- Follow-up: hands requests the bot does not answer to staff as leads.

### 3.2 Per-source collections and merged retrieval

One persistent directory holds three collections, one per source. When only the specs change, only the `product_spec` collection is dropped and rebuilt; the operational script supports per-source re-indexing as `embed_single.py product_spec | docx | all`. At query time, each collection returns k results via `similarity_search_with_score(query, k)`, and the union is sorted by distance to keep the top k. All collections share the same model and normalization, so distances are directly comparable.

### 3.3 Chunking and enrichment

| Strategy | Description |
|---|---|
| `by_product` | One section = one document |
| `recursive_N` | RecursiveCharacterTextSplitter, N = 300 – 1,500 in steps of 10 (121 values), overlap = max(10, 0.1N). Sections no longer than N are left intact |

| Enrichment | Prepended to each chunk | Cost |
|---|---|---|
| `none` | Minimal `[category] product` header | none |
| `enriched` | `[source]`, `[category]`, `[product]`, `[section]`, `[doc type: description]` tags | none |
| `contextual` | enriched tags + a 2–3 sentence `[context]` written by an LLM | one LLM call per chunk |

`contextual` follows Anthropic's Contextual Retrieval. Sections from the same source, category, and product are concatenated as document context (500 chars per section, 2,000 chars total), and the LLM describes what the chunk covers, its role in the document, and useful search keywords. gpt-4o-mini (temperature 0, max_tokens 200) is called from 5 threads, and results are cached on disk keyed by `md5(document context + chunk)` so repeated runs make no new calls. Chunks whose call fails keep their original text.

### 3.4 PII masking and streaming restoration

Korean pattern recognizers were added to Presidio's `AnalyzerEngine`: phone numbers (delimited format at score 0.9, bare 10–11 digit strings at 0.7), emails (0.9), and names (2–4 Hangul characters followed by an honorific or job title via lookahead, 0.7). Detected spans are replaced with `<ENTITY_i>` and a restoration mapping is returned. Replacement runs in reverse start order so earlier offsets stay valid.

`fordocker/pii_handler.py`
```python
    results = detect_pii(text)
    
    if not results:
        return text, {}
    
    # 복원용 매핑 생성
    mapping = {}
    masked_text = text
    
    # 위치 역순으로 정렬 (뒤에서부터 치환해야 위치 안 꼬임)
    sorted_results = sorted(results, key=lambda x: x.start, reverse=True)
    
    for i, result in enumerate(sorted_results):
        original = text[result.start:result.end]
        placeholder = f"<{result.entity_type}_{i}>"
        
        mapping[placeholder] = original
        masked_text = masked_text[:result.start] + placeholder + masked_text[result.end:]
    
    return masked_text, mapping
```

The LLM echoes placeholders in its answer, and the server maps them back. In streaming, a token stream that has only delivered `<PHONE_` cannot be restored yet, so the server keeps buffering while the buffer ends with an unclosed tag and otherwise restores and flushes the buffer. With an empty mapping, tokens pass through unbuffered.

`fordocker/server.py`
```python
    pii_pattern = re.compile(r'<[A-Z_]+_\d+>')
    partial_pattern = re.compile(r'<[A-Z_]*$')  # 불완전한 태그 감지
    
    buffer = ""
    
    for chunk in llm.stream(prompt):
        buffer += chunk
        
        # 불완전한 PII 태그가 버퍼 끝에 있는지 확인
        if partial_pattern.search(buffer):
            # 버퍼에 불완전한 태그가 있으면 계속 축적
            continue
        
        # 완전한 태그들을 복원하고 출력
        restored = unmask_pii(buffer, pii_mapping)
        yield restored
        buffer = ""
```

Retrieval used the raw question, on the reasoning that the embedding model runs locally inside the server and nothing leaves it.

### 3.5 Prompt rules

Separate Korean and English prompts were selected by the session's language setting.

| Rule | Behavior |
|---|---|
| Grounding first | Answer if the documents cover it; otherwise say so and suggest a consultation request |
| Clarify the type | For product families that split into two types by usage, ask for the type first if missing, and explain the difference in one line if the customer does not know |
| Route to staff | Pricing, safety and technical documents, and unrelated inquiries go to a consultation request |
| Competitor brands | Ask about the usage environment without naming own products or saying "replacement" or "compatible"; includes counterexamples so ordinary company names are not mistaken for brands |

Output rules specified no Markdown (the chat UI does not render it), product names in brackets, one attribute per line, and expanded abbreviations.

### 3.6 Multi-turn session manager

Under a `BaseSessionManager` interface sit three implementations: Redis (a JSON list per key, `setex` refreshing a 24-hour TTL), database (queries the conversation table), and memory (a dict, purging expired sessions on creation and counting). With `SESSION_TYPE=auto`, the factory tries a Redis `ping` first, then the database, then memory. All three keep the last 10 exchanges (20 messages) and render them as `customer: ... / bot: ...` text for the history slot in the prompt.

### 3.7 Serving and leads

- **LLM.** Azure OpenAI gpt-4o-mini (temperature 0.1, max_tokens 1024) sits behind a `BaseLLM` (invoke, stream) interface so a local model can be swapped in.
- **WebSocket.** The server emits `session` → `sources` → `start` → `chunk`* → `end`; besides questions, the client sends `set_language`, `clear`, and `submit_lead` actions.
- **Safeguards.** 2,000-character question limit and a per-session limit of 30 requests per minute. In production mode the server refuses to start without an admin password and explicit CORS origins.
- **Leads.** Form fields are stored as a lead and linked to the session's conversation. Status moves only along `new → contacted → converted → closed`. In the admin view, the LLM summarizes the conversation in 2–3 sentences and assigns one of five inquiry types (A–E).

## 4. Experiments

### 4.1 Setup

| Item | Setting |
|---|---|
| Embedding models | bge-m3 (1024d), multilingual-e5-large (1024d), ko-sroberta-multitask (768d) |
| Chunking | by_product + 121 recursive sizes = 122 |
| Enrichment | none, enriched (default), contextual (opt-in) |
| Combinations | 3 × 122 × 2 = 732 by default, 1,098 with contextual |
| Embedding runtime | CPU, normalized, batch 32 |
| Test set | 20 questions (question, reference answer, target product): 11 product description/recommendation, 5 property values, 4 company/dealer |

### 4.2 Results

Per-combination metric files and indexes are excluded from the repository, so Hit@5 and MRR@5 values are not reported here. The table below shows chunk statistics from running the repository's chunking code on all catalog and spec sections (chunk counts shown as a ratio to by_product).

| Chunking | none: chunks | none: mean chars | enriched: chunks | enriched: mean chars |
|---|---|---|---|---|
| by_product | ×1.00 | 329 | ×1.00 | 419 |
| recursive_300 | ×2.18 | 153 | ×2.46 | 172 |
| recursive_500 | ×1.02 | 322 | ×1.08 | 387 |
| recursive_800 | ×1.01 | 326 | ×1.01 | 414 |
| recursive_1000 | ×1.00 | 328 | ×1.00 | 417 |
| recursive_1500 | ×1.00 | 329 | ×1.00 | 419 |

Enriched tags add about 90 characters per chunk, so the same N produces more splits.

### 4.3 Ablation

| Axis | Observation | Verdict |
|---|---|---|
| Effective range of N | Of 121 values, only 44 (none) and 47 (enriched) produce distinct chunk sets; from N ≥ 1,010 (none) and N ≥ 1,090 (enriched) the output equals by_product | 35–41% of the sizes at the top of the range are duplicates |
| Test set resolution | 20 questions, so one question = 5 points of Hit@5 | Picking the best of 122 is hard to separate from noise |
| Hit criterion | A substring match of the product name anywhere in the text counts as a hit | Chunks that list several products are over-credited |
| RAGAS generation metrics | The reference answer is passed in the `answer` field instead of the bot's output | Faithfulness and answer relevancy are not meaningful |
| k in experiment vs. serving | top-5 in experiments, k=3 in serving | Experiment scores do not represent served quality |
| Contextual cost | One LLM call per chunk; the hash cache removes re-run cost | Excluded from the default run |

Only about 11% of catalog sections exceed 400 characters, and every spec section is 400 or shorter. A few sizes between 300 and 1,000 plus a larger test set would be more informative for the same compute. The company-info source could not be included in this analysis.

## 5. Limitations and future work

1. **PII in history.** The question is masked before it reaches the LLM, but the session stores the raw text and that history goes into the prompt. On the WebSocket path the user message is appended before generation, so the raw text is sent in the same turn. The admin summary also sends the raw conversation to the LLM. History and summary inputs should be stored masked.
2. **Restoration regex.** `<[A-Z_]*$` only catches tags cut inside the entity name, such as `<PHONE_`. If a chunk boundary falls between `<PHONE_NUMBER_0` and `>`, the incomplete tag is emitted unrestored. The pattern should cover the numeric tail, e.g. `<[A-Z_]*\d*$`.
3. **Experiment and serving index diverge.** The serving code loads the single collection built by the first crawl-based pipeline with k=3, while per-source collections, enrichment, and distance merging exist only in the experiment pipeline. Both should be wired to the same build code.
4. **Test set.** Questions should be sampled from real consultation logs, labeled with target chunk IDs, and grown past 100. The 1,000 template-generated questions are too uniform in wording to serve as more than a supplement.
5. **Prompt length and duplicate retrieval.** Formatting rules in the prompt can conflict with other rules; mechanical ones such as stripping Markdown and expanding abbreviations can move to post-processing. Each WebSocket turn also retrieves twice (for the source list and for generation), and reusing the first result would cut embedding calls.

## References

- Tistory development log: https://mynews76154.tistory.com/13
- Embeddings: BAAI/bge-m3, intfloat/multilingual-e5-large, jhgan/ko-sroberta-multitask
- LLM: Azure OpenAI gpt-4o-mini
- Frameworks: ChromaDB, LangChain, FastAPI, Redis
- PII: Microsoft Presidio
- Evaluation: RAGAS
- Technique: Anthropic, "Contextual Retrieval"
