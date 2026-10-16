---
title: "LawKeeper: Multi-turn Legal Consultation RAG with a Fixed Case Context and Query Rewriting for Follow-up Retrieval"
description: "A legal consultation RAG chatbot that splits statutes, precedents, terms-of-service clauses and consultation Q&A into five ChromaDB collections, and retrieves for follow-up questions with a rewritten query built from a fixed case summary plus the current question."
date: 2026-10-16
lang: en
key: lawkeeper
tags: [RAG, Multi-turn RAG, Query Rewriting, ChromaDB, Ko-SBERT, GPT-4o-mini, Legal AI]
tistory: https://mynews76154.tistory.com/24
---

## Abstract

LawKeeper is a legal consultation RAG chatbot: the user describes a legal situation, and the system retrieves statutes, precedents, terms-of-service clauses and consultation cases to produce grounded advice. It was built by a team of four and deployed as a web service for desktop and mobile. Source data in three formats (JSON, CSV, PDF) was normalized into a common `{id, content, metadata}` schema, embedded with `jhgan/ko-sbert-nli` into five collections, and the top-5 hits per collection are passed to GPT-4o-mini. The core problem was retrieval for follow-up questions that carry no case information. Across ten versions of the dialogue loop, the design converged on summarizing and freezing the case description once it is judged sufficient, then retrieving each follow-up with `summarize(fixed case + current question)`. The repository contains no quantitative evaluation set, so the comparisons here are limited to design differences that can be verified in code.

## 1. Problem

**Task.** Given a layperson's description of a legal situation, generate advice grounded in statutes and precedents, and keep answering follow-up questions within the same case context.

| Stage | Input | Output |
|---|---|---|
| Initial consultation | Case description (may accumulate over several turns) | One clarifying question if insufficient; otherwise structured advice with eight sections |
| Follow-up | Follow-up question + fixed case summary + dialogue history | A direct answer to the question |

**Evaluation.** The repository has no evaluation set, gold document labels or retrieval metric code. Versions were accepted by typing in queries and reading the results (per the development log). This report therefore compares versions at the code level against two design goals: whether retrieved context mixes several evidence types (goal 1, source diversity), and whether the follow-up retrieval query always contains the case facts (goal 2, context retention).

**Constraints.** Generation used `gpt-4o-mini` in every version, embeddings were computed locally with SentenceTransformer, and storage was ChromaDB `PersistentClient`. ChromaDB metadata accepts scalar values only. The service ran on Google Cloud Run with logs in BigQuery (per README).

## 2. Data

Sources were judgment, terms-of-service and situation-based precedent JSON from public datasets; CSV scraped with Selenium from the Korea Legal Aid Corporation's legal counsel and cyber counsel boards; public QA JSON (instruction format, easylaw); and a PDF of six codes (Constitution, Civil Act, Criminal Act, Commercial Act, Civil Procedure Act, Criminal Procedure Act). Regardless of format, body text was separated from metadata and each document was saved as one JSON file.

| Collection | Source | Body | Main metadata | Chunking |
|---|---|---|---|---|
| `legal_laws` | Six-code PDF | One article | Law name, article | Per article |
| `legal_rag_docs_kosbert` | Situation-based precedents and rulings | Tagged sections: `[판시사항]` (holdings), `[판결요지]` (summary of judgment), `[요약]`, `[질문]/[답변]` | Case no., court, date, keywords, cited articles, cited precedents | None |
| `legal_cases` | Court judgments | `취지`, `사실`, `판단` (purpose, facts, reasoning) joined | Case no., date, case name, court, case type | None |
| `terms_clauses` | Terms-of-service clauses | Field + clause + favorable/unfavorable verdict + illegality basis + related laws | Field, FTC ruling, verdict | None; docs under 100 chars dropped |
| `legal_qa_rag_docs` | Scraped CSV, public QA | `Q: ... A: ...` or `### Instruction / ### Output` | Original columns, source_file | 400 chars, overlap 100 |

Apart from the 8,997 terms JSON files named in the terms ingestion script's log message, document counts are not recorded in the repository.

Preprocessing had four branches. For judgments and terms, numeric codes were mapped to text from the dataset spec (`caseField` "1" → "civil", `clauseField` "5" → "lease contract") and terms keys were renamed to Korean (`02_`). Situation-based precedents were merged 1:1 by file name across source (TS/VS) and label (TL/VL) folders, turned into tagged body text, and their cited articles and precedents were moved into metadata (`04_`, `05_`). QA JSON was branched by structure, and each CSV row became one document (`07_`). Embedding raw codes would leave words like "lease contract" absent from the document, so it could never match a user query.

## 3. Method

### 3.1 Pipeline

```text
[Offline ingestion]
 ├─ Judgment/terms JSON ─── code → text mapping ─────────────┐
 ├─ Precedent JSON ──────── source/label merge → tagged body ┤
 ├─ Consultation Q&A ────── Q/A documents → 400/100 chunks ──┤
 └─ Six-code PDF ────────── font-based article split → clean ┘
                               ↓
     {id, content, metadata} → ko-sbert-nli → 5 ChromaDB collections

[Online dialogue]
 ├─ Initial mode: accumulate case description → sufficiency check
 │    ├─ insufficient → one clarifying question
 │    └─ sufficient  → freeze case summary → 5 collections × top-5 → initial advice
 └─ After advice: summarize(fixed case + question) → 5 collections × top-5
                  + history summary → follow-up answer
```

- Sufficiency check: GPT returns `YES` or a single clarifying question.
- Case freezing: the summary at the moment the check passes becomes the anchor for every later retrieval.
- Query rewriting: each follow-up is merged with the fixed case and re-summarized before retrieval.

### 3.2 Metadata normalization

Keywords, cited articles and cited precedents are lists, which ChromaDB metadata rejects. Right before ingestion they were serialized to JSON strings and `None` was replaced with an empty string; the precedent and Q&A loaders share the same `flatten_metadata` function. The chunked loader (`09_`) copies the parent metadata onto every chunk and uses `<source file>_chunk{i}` as the ID so each chunk can be traced back to its document.

### 3.3 Article boundary detection in the code PDF

Start and end pages per law were kept in a dictionary (e.g. Civil Act, pp. 21-132) and only those ranges were processed. A string like "Article 750" also appears inside body text as a citation, so regex alone cannot tell a heading from a reference. Only article heading spans use a specific font, size and flag combination, so the text pattern was combined with font conditions.

`code/19_법전 json ocr 코드 수정 제거 치환 추가 - 최종.py`
```python
                    is_title_candidate = (
                        raw_text.startswith("제") and
                        "조" in raw_text and
                        font == "H2gtrE" and
                        abs(size - 7.79) < 0.05 and
                        flags == 4
                    )
```

When a heading broke mid-way (`제3조(정의` with `)` in the next span), following spans were appended until the closing parenthesis appeared; chapter and section headings were dropped from the body. Post-processing removed parenthesized Chinese characters, textbook footers and page numbers like `- 301 -`, and replaced circled numerals ①-㊿ with "1항"-"50항" (paragraph 1-50). Users never type circled numerals, so the document notation was aligned with how queries are written. Spacing broken by PDF extraction was then corrected with PyKoSpacing.

### 3.4 Separate collections and a single embedding model

With everything in one collection, consultation Q&A, whose style is closest to user queries, tends to dominate the top results and crowd out the statutes and precedents the prompt asks for. Taking top-K per collection guarantees that all five evidence types reach the LLM, and each document is prefixed with its collection name, e.g. `[출처: legal_cases]` (source: legal_cases), so the model can tell evidence types apart.

The early ingestion scripts used a different model per collection: `jhgan/ko-sbert-nli` for judgments, `dragonkue/snowflake-arctic-embed-l-v2.0-ko` for terms, and `all-MiniLM-L6-v2` for precedents. The query is embedded once and used against every collection, so mixed models put query and documents in different vector spaces, and mismatched dimensions make the query fail outright. The final code uses `jhgan/ko-sbert-nli` throughout, and the precedent collection was renamed `legal_rag_docs_kosbert`.

### 3.5 Retrieval: from a distance threshold to per-collection top-K

The first implementation (`13_`) fetched three hits per collection and kept those scoring at least 0.5.

`code/13_크로마디비 4개 RAG 구현 코드 임계값(쓰레스홀스) 0.5 프롬프트 적용.py`
```python
threshold = 0.5  # ✅ 유사도 임계값 설정
all_docs = []

for collection in collections:
    results = collection.query(query_embeddings=[query_embedding], n_results=N_RESULTS)
    docs = results["documents"][0] if results["documents"] else []
    scores = results["distances"][0] if results["distances"] else []

    for doc, score in zip(docs, scores):
        if score >= threshold:
            all_docs.append(doc)
```

ChromaDB `distances` are smaller for closer documents, so this filter discards the nearest hits. The default `hnsw:space` is L2, and only the terms collection specified cosine, so the scale also differed by collection. After a 0.3 threshold (`14_`, `15_`) and a score-printing debug version (`17_`), the threshold was removed from `22_` onward. top-K went from 3 to 10 and finally 5, so up to 25 documents enter the prompt per retrieval.

### 3.6 Multi-turn state and follow-up retrieval

**Sufficiency check.** In initial mode, inputs are accumulated and a summary is sent to GPT to decide whether advice can start. The prompt lowers the bar ("YES if a legally problematic fact is clear, even if brief") to avoid repeated clarifying questions. If the input contains negative cues such as "I don't know" or "none" and exceeds 30 words, it is accepted as `YES` without asking GPT.

**Case freezing and query rewriting.** When the check passes, the accumulated description is summarized and frozen; later follow-ups are not appended to it. Embedding "How long does a criminal complaint take?" as-is retrieves documents close to "complaint" and "duration" but unrelated to the case, so the fixed summary and the question are merged and re-summarized into two or three lines that serve as the query. Earlier follow-ups are excluded from the query.

`code/28_FAST API.py`
```python
    if advice_given:
        search_query = summarize_context(fixed_case_context + "\n" + question)
        document_context = search_documents(search_query, N_RESULTS)
        answer = generate_followup_rag_response(question, fixed_case_context, document_context, chat_history)
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        return answer
```

**History summary.** For generation, the last ten messages are truncated to 300 characters each, summarized by GPT, and passed along with the case summary, retrieved documents and question. The advice turn makes up to four LLM calls (summary, sufficiency check, case summary, advice); each follow-up turn makes three (query rewrite, history summary, answer).

### 3.7 Prompts and generation settings

The initial advice prompt requires a situation recap followed by eight fixed sections: legal assessment, applicability of precedents, practical advice, realistic options, immediate actions, cautions, summary, and questions to prepare for a lawyer. The follow-up prompt drops the fixed format and asks for a direct answer; an instruction not to repeat the advice format has been present since `16_`.

| Call | temperature | max_tokens |
|---|---|---|
| Case / query summary | 0.3 | 100 |
| History summary | 0.2 | 100 |
| Sufficiency check | 0.7 | 100 |
| Initial advice | 0.5 | 1500 |
| Follow-up answer | 0.6 | 800 |

### 3.8 Serving

`28_FAST API.py` is a module exposing `ask_ai(question)` for the web server plus a CLI loop. The README and slides describe Cloud Run deployment, BigQuery logging, caches, parallel retrieval and a session-managing service class, but the server code is not in the repository.

## 4. Experiments

### 4.1 Setup

- Compared: ten dialogue-loop versions (`13`-`16`, `22`-`26`, `28`) and the retrieval debug version `17`.
- Fixed: `gpt-4o-mini`, `jhgan/ko-sbert-nli`.
- Protocol: no quantitative evaluation set. Versions are compared by tracing in code how the retrieval query and results are built and delivered to the LLM.
- Ingestion: embedding batch 32 (judgments), insert batch 500 (precedents, Q&A), Q&A chunking 400/100. The terms collection was created with `hnsw:space=cosine`, `M=32`, `construction_ef=400`, `search_ef=50`.

### 4.2 Results

| Component | Final setting (`28_`) |
|---|---|
| Collections | 5 |
| Retrieval | top-5 per collection, no threshold, up to 25 documents |
| Initial query | Summary of the case description that passed the sufficiency check |
| Follow-up query | `summarize_context(fixed_case_context + "\n" + question)` |
| RAG on follow-ups | Always (no routing step) |
| Generation input | Case summary + history summary + retrieved docs + question |

Goal 1 is met structurally by per-collection top-K, and goal 2 by always including the fixed case summary in the follow-up query. No retrieval accuracy or answer quality numbers exist in the repository.

### 4.3 Ablation

| Version | Retrieval query | Filter / top-K | Follow-up handling | Behavior verified in code |
|---|---|---|---|---|
| 13 | Raw input each turn | distance ≥ 0.5 / 3 | Full history + retrieved docs | Collections deleted and recreated on every run, so empty; filter direction inverted |
| 14·15 | First situation description | distance ≥ 0.3 / 3 | Full history | Retrieved docs never added to `messages`, so RAG is effectively off |
| 16 | Accumulated description | distance ≥ 0.5 / 3 | RAG only for the first advice | Sufficiency check added; collection deletion removed |
| 22 | Accumulated description | none / 3 | Same as 16 | Threshold removed; GPT general knowledge allowed when no docs |
| 23 | Accumulated description | none / 3 | Same as 16 | "Strictly based on reference documents" prompt |
| 24 | Raw follow-up | none / initial 10, follow-up 5 | GPT routes RAG yes/no | Follow-ups appended to the case description; precedent collection path points at the QA folder |
| 25 | Fixed summary + question | none / initial 10, follow-up 5 | Routing kept; history summarized by GPT | Case frozen; earlier follow-ups excluded |
| 26 | summarize(fixed summary + question) | none / 5 | Routing removed; always RAG | Summary prompt asks to surface legal issues and drop product names and prices |
| 28 | summarize(fixed summary + question) | none / 5 | Same as 26 | Summary prompt reverted to "2-3 factual lines"; `ask_ai` added |

The grounding policy of the generation prompt moved from 22 (general knowledge allowed) to 23 (documents only) to 26 (documents first, never end a section with "not applicable"). When retrieved documents are unrelated to the case, a strict prompt yields empty answers and a loose one yields ungrounded answers, so this axis cannot fix a retrieval problem. The changes whose effect is visible in the code structure are anchoring the query on the case (25) and rewriting it into a retrieval-oriented sentence (26). The RAG routing step in 24 and 25 was removed in 26: a wrong `NO` leaves the answer with no evidence at all, and the router itself costs an extra GPT call.

## 5. Limitations and future work

1. **No quantitative evaluation.** Bugs like the one in 14 and 15 were invisible from the answers alone. A fixed query set with gold document labels should be used to measure per-collection Recall@K for each version, and the retrieval query, results and final prompt should be logged every turn.
2. **References to earlier follow-ups are lost.** The query uses only "fixed case + current question", so a question like "that article you mentioned" loses its referent. A condense-question step that rewrites the history plus current question into a standalone query is needed.
3. **Summary length vs. token limit.** The history summary prompt asks for up to 600 characters but `max_tokens=100` can cut it off; the query summary has the same limit. The issue-focused rewrite prompt from 26 that disappeared in 28 should be restored.
4. **Global state.** `ask_ai` keeps its state in module globals, so calling it directly from a web server would make all users share one consultation. Whether the server code separated sessions cannot be verified from the repository.
5. **Persona and assertive phrasing.** The final prompt uses a "lawyer with 15 years of experience" persona and tone examples such as "Legally, this clearly constitutes the crime of X." This runs opposite to the "do not assert illegality" rule in 13 and conflicts with the goal of evidence-based advice.

## References

- Code: [github.com/tjehdgus/lawkeeper_rag_openAI](https://github.com/tjehdgus/lawkeeper_rag_openAI)
- Development log (Tistory, Korean): [LawKeeper development log](https://mynews76154.tistory.com/24)
- Embedding: jhgan/ko-sbert-nli (Sentence-BERT)
- Generation: OpenAI GPT-4o-mini
- Vector DB: ChromaDB (HNSW)
- Preprocessing: PyMuPDF, PyKoSpacing, LangChain RecursiveCharacterTextSplitter, Selenium
