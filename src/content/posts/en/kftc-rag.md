---
title: "RAG over Korea Fair Trade Commission Decisions: Hybrid Retrieval, Dual-Reranker RRF, and EXAONE 3.5 LoRA Generation"
description: "A RAG system that returns five evidence chunks and an answer over 500 KFTC decisions (31,877 chunks), built under a local-LLM-only (≤8B), no-rechunking, 30-second rule. On a 278-question internal set: R@5 0.6930, MRR 0.8007, total score 67.76."
date: 2026-10-16
lang: en
key: kftc-rag
tags: [RAG, Hybrid Retrieval, Reranker, RRF, Contextual Chunking, EXAONE, LoRA]
tistory: https://mynews76154.tistory.com/1
---

## Abstract

For Track 2 of the 2nd Fair Trade AI and Data Competition (April to May 2026, team of two), we built a RAG system over 500 decisions of the Korea Fair Trade Commission (KFTC) that returns five evidence `chunk_id`s and a natural-language answer per query. The rules prohibited rechunking, limited generation to local LLMs of 8B parameters or fewer, and capped latency at 30 seconds per question, so retrieval quality had to come from the indexed text (context prefixes, normalization, table-to-text) rather than chunk boundaries. Retrieval fuses hybrid BM25 + dense results from two embedding models with RRF, then fuses the top-5 lists of two pretrained rerankers with a second RRF. Generation uses EXAONE 3.5 7.8B (4-bit NF4) with a LoRA on seven attention and MLP modules, continued from a 1,500-example adapter on 5,000 new examples. On our 278-question internal set the system reached R@5 0.6930, MRR 0.8007, BERTScore F1 0.7830, token F1 0.4000, and a total score of 67.76.

## 1. Problem

The input is a natural-language query; the output is five evidence `chunk_id`s and an answer. The grader sends queries to `POST /predict` on a Docker container. Retrieval and generation are worth 50 points each.

```text
R@5   = |top5 ∩ gold| / |gold|        (averaged over queries)
MRR   = 1 / rank of first gold chunk in top5, 0 if none
Final = 0.35*R@5 + 0.15*MRR + 0.30*BERTScore_F1 + 0.20*TokenF1   (x100)
```

In our own measurements BERTScore used `klue/roberta-base`. The constraints:

| Rule | Penalty |
|---|---|
| Exactly 5 `chunk_id`s, no duplicates | Retrieval score 0 |
| Only original `chunk_id`s (no rechunking) | That chunk counted wrong |
| ≤ 30 s per question | Whole question scored 0 |
| Generation LLM ≤ 8B, no external APIs, offline | Disqualification |

The grading environment had two A100 80G GPUs. Embedding model size was unrestricted.

## 2. Data

| Item | Size | Notes |
|---|---|---|
| Decision corpus | 500 decisions, 31,877 chunks | IDs `DOC-{UUID}-CH-{n}` |
| Table chunks | 10,609 (33.3%) | ≥ 5 `\|` characters in `raw_text` |
| Internal eval set `qa_dataset.json` | 278 questions | 2.04 gold chunks per question on average (1 to 3), reference answers 110 chars on average |
| Training QA v1 `train_qa.json` | 1,500 | Generated with GPT-4o-mini, answers 88 chars on average |
| Training QA v6 `train_qa_v6.json` | 5,000 | GPT-4o-mini, three personas, gold-chunk layout 30% single / 40% adjacent / 30% scattered |
| Official test set | 200 questions | Hidden |

The original `raw_text` was kept untouched; only the `text` field used for indexing was rebuilt.

1. **Context prefix.** Each chunk is prefixed with `[violation type] [document title] [section] [header]`.
2. **Regex normalization.** Dates `2017.4.27.` become `2017년 4월 27일`, amounts get a rounded Korean-unit gloss appended, and anonymization glyphs (`●○☆△`, `ㅇㅇㅇ`) become `[비공개]`.
3. **Table-to-text.** Table chunks were rewritten as prose with GPT-4o-mini, and only the 1,524 conversions (14.4%) that passed validation were applied. This ran once at indexing time, so no external API is called at inference.

The prefix was motivated by a failure analysis (`analysis/analyze_failures.py`): queries whose gold chunk was shorter than 100 characters failed 90% of the time, 46.4% of wrong hits were neighbors of the gold chunk (CH±1 or ±2), and 31.8% of chunks were table-shaped.

## 3. Method

### 3.1 Pipeline

```text
query
 ├─ BGE-m3-ko FAISS      + BM25   (min-max linear fusion)
 └─ Arctic-embed-ko FAISS + BM25   (min-max linear fusion)
        ▼
 RRF over the two hybrid rankings (k=60) → 5 x pool candidates
        ├─ mxbai-rerank-large-v2  30 candidates → top 5
        └─ bge-reranker-v2-m3     40 candidates → top 5
        ▼
 RRF over the two top-5 lists (k=60) → 5 chunk_ids
        ▼
 EXAONE 3.5 7.8B (4-bit NF4) + LoRA v6_c, greedy, 128 tokens → answer
```

- Hybrid: each embedding retriever min-max normalizes its BM25 and dense scores and combines them linearly.
- Ensemble and reranker fusion: `score = Σ weight / (k + rank + 1)`. RRF was used for the rerankers because their score scales differ and only ranks are comparable (`scripts/rrf_merge_rerankers.py`).
- Generation: the top five chunks are passed as a numbered context, `[1]` to `[5]`.

| Component | Owner |
|---|---|
| Embedding model comparison and selection, reranker fine-tuning (v2 to v6) | Teammate |
| Failure analysis, Contextual Chunking, index normalization, table-to-text, reranker RRF fusion | Me |
| LoRA training data, EXAONE LoRA training, generation evaluation | Me |

### 3.2 Contextual Chunking

This is a lightweight take on Anthropic's Contextual Retrieval that uses existing metadata instead of an LLM call. The same function is used by both the FAISS build (`shared/data/loader.py`) and the BM25 build (`shared/indexing/build_bm25.py`), so both indexes see identical text. The violation-type field is an aggregated string such as `A; A; B`, so it is deduplicated first.

`shared/data/chunk_formatter.py`
```python
    parts: List[str] = []
    vt = normalize_violation_types(violation_type)
    if vt:
        parts.append(f"[{vt}]")
    if doc_title:
        parts.append(f"[{doc_title}]")
    if section:
        parts.append(f"[{section}]")
    if header:
        parts.append(f"[{header}]")
    prefix = " ".join(parts)
    return f"{prefix} {page_content}" if prefix else page_content
```

### 3.3 Validating table-to-text

If a conversion alters an amount or a date, the retrieved evidence is wrong even when retrieval is right. A conversion is accepted only if at least 80% of the numeric tokens and 60% of the Korean words (2+ characters) survive; failures and API errors fall back to the original text.

`scripts/gpt_convert_tables.py`
```python
def validate_conversion(original: str, converted: str) -> bool:
    orig_nums = set(re.findall(r"[\d,]+", original))
    conv_nums = set(re.findall(r"[\d,]+", converted))
    if orig_nums:
        num_keep_ratio = len(orig_nums & conv_nums) / len(orig_nums)
        if num_keep_ratio < 0.8:
            return False

    orig_words = set(re.findall(r"[가-힣]{2,}", original))
    conv_words = set(re.findall(r"[가-힣]{2,}", converted))
    if orig_words:
        word_keep_ratio = len(orig_words & conv_words) / len(orig_words)
        if word_keep_ratio < 0.6:
            return False

    return True
```

### 3.4 EXAONE 3.5 LoRA

EXAONE names its MLP projections `c_fc_0`, `c_fc_1`, and `c_proj` (the counterparts of LLaMA's gate/up/down_proj), so the training script checks the targets against `named_modules()` and warns if any are missing. Trainable parameters were 13,631,488 (0.174%) for the four attention modules and 41,943,040 (0.534%) with the MLP modules added. Training examples use the EXAONE chat format (system instruction, reference chunk plus question, answer), with a fixed instruction asking for one or two sentences of about 100 characters and no markdown or meta comments. The final adapter, v6_c, loads the seven-module v1 adapter as trainable and continues training on the 5,000 v6 examples.

`training/lora/mlp_finetune_exaone.py`
```python
    if resume_adapter:
        print(f"[4/5] 기존 LoRA 어댑터 로드 (continued FT): {resume_adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
        model.print_trainable_parameters()
    else:
        print(f"[4/5] LoRA 어댑터 적용 (target={len(TARGET_MODULES)}개: attention 4 + MLP 3)")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
```

The continued-training path does not build a new `LoraConfig`, so r, alpha, and dropout are inherited from the starting adapter (16, 32, 0.05).

## 4. Experiments

### 4.1 Setup

- Evaluation: the 278-question internal set as a single holdout. Development and measurement on one RTX 5070 Ti (16 GB).
- Retrieval: `top_k 5`, chunk-level scoring, `5 x pool` reranker candidates.
- LoRA, shared: r 16, alpha 32, dropout 0.05, batch 1 x grad accumulation 8, 30 warmup steps, max_seq 1536, `paged_adamw_8bit`, fp16.
- v1 and seven-module: 2 epochs, lr 2e-4, 1,500 examples (40 min and 48.4 min). v6_c: 1 epoch, lr 1e-4, 5,000 examples, about 122 min.
- Inference: greedy, max_new_tokens 128, zero-shot, basic post-processing (strips markdown, numbering, and meta phrases).

#### Evaluation protocol

- Training QA (v1, v6) was generated separately; no question text overlaps with the eval set.
- The v6 generator excludes the eval set's 566 gold chunks from its seed pool (`scripts/generate_train_qa_v6.py`).
- Few-shot examples are drawn only from `train_qa.json` (`--few_shot_qa_path`).
- Upper-bound runs that inject the gold violation type are isolated behind `--use_oracle_vt_filter` and `*_oracle.json` file names and are not used for selection.

These rules were in force from April 30 onward. Before that, eval questions had entered training in two places: the contrastive embedding fine-tune (mine, which used all 278 eval questions as triplet queries) and the reranker fine-tune (teammate's, trained on 222 of them). The 72.96 measured in that state was voided and the reranker reverted to the pretrained model.

### 4.2 Results

| Stage | Metric | Value | Points |
|---|---|---|---|
| Retrieval | R@5 / MRR | 0.6930 / 0.8007 | 36.27 / 50 |
| Generation | BERTScore F1 / token F1 | 0.7830 / 0.4000 | 31.49 / 50 |
| Total | | | 67.76 / 100 |

The share of questions with at least one gold chunk in the top 5 was 0.9317, and generation took 5.66 s per question on average. Score progression from the experiment log (voided measurements excluded):

| Date | Change | Retrieval | Generation | Total |
|---|---|---|---|---|
| 4/24 | Baseline (no LoRA) | 31.38 | 28.53 | 59.91 |
| 4/27 | + LoRA v1 | 32.92 | 30.90 | 63.82 |
| 4/29 | + seven-module LoRA | 32.92 | 31.12 | 64.04 |
| 4/29 | + embedding-side work (teammate) | 36.02 | 31.12 | 67.14 |
| 5/3 | Back to pretrained reranker | 35.95 | 31.12 | 67.07 |
| 5/4 | Normalization + dual-reranker RRF | 36.27 | 31.12 | 67.39 |
| 5/5 | + LoRA v6_c | 36.27 | 31.49 | 67.76 |

The 32.92 retrieval score on 4/27 to 4/29 was measured with the embedding fine-tune that had seen eval questions.

### 4.3 Ablations

**Retrieval (1), early setup (`top_k 10`, bge reranker, pool 8).**

| Configuration | R@5 | MRR | Time (s) |
|---|---|---|---|
| 5-model ensemble, oracle | 0.5402 | 0.6919 | 29.38 |
| 2-model (bgem3ko + arctic), oracle | 0.5420 | 0.6955 | 2.98 |
| 2-model + Contextual, oracle | 0.5947 | 0.7349 | 4.65 |
| 2-model + Contextual (FAISS + BM25) | 0.5815 | 0.7350 | 3.33 |
| 2-model + Contextual, pool 20 | 0.5647 | 0.7307 | 7.28 |
| 2-model + Contextual + violation-type classifier filter | 0.4844 | 0.6148 | 11.82 |
| 2-model + Contextual + contrastive embedding FT (mine, voided) | 0.5815 | 0.7291 | |

Five models matched two models on R@5 at roughly ten times the latency. Contextual Chunking added 5.27 points of R@5 under the same two-model oracle condition. The +8.45 points in the commit message was measured against the five-model, unfiltered run (0.4934), so it also includes the change in model count. The classifier filter (62.95% accuracy) lost 9.71 points because a wrong prediction removes the gold chunk from the candidates.

**Retrieval (2), mxbai candidate pool (`top_k 5`).**

| pool | 4 | 6 | 8 | 10 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| R@5 | 0.6871 | 0.6841–0.6960 | 0.6823 | 0.6793 | 0.6703 | 0.6727 | 0.6703 |
| MRR | 0.8001 | 0.7944–0.8052 | 0.7964 | 0.8005 | 0.7931 | 0.7919 | 0.7844 |
| Time (s) | 0.92 | 1.46–1.71 | 1.72 | 2.22 | 2.82 | 3.45 | 4.01 |

The pool-6 column is the range over four runs of the same configuration. Pools 4 and 8 fall around that range; from 10 upward R@5 declines.

**Retrieval (3), reranker setup and normalization.**

| Configuration | R@5 | MRR |
|---|---|---|
| mxbai pool 6, before normalization (best run) | 0.6960 | 0.7944 |
| mxbai pool 6, after normalization | 0.6924 | 0.7978 |
| bge pool 8, after normalization | 0.6745 | 0.7809 |
| mxbai + bge RRF, after normalization (final) | 0.6930 | 0.8007 |
| Fine-tuned reranker v6 (teammate), pool 8 | 0.6487 | 0.7708 |
| 3-model (+ Qwen3-Embedding INT8, teammate), mxbai pool 6 | 0.6805 | 0.7936 |

Normalization (R@5 -0.36, MRR +0.34 points) and dual RRF (MRR +0.29 points over mxbai alone) are both within the pool-6 run-to-run spread.

**Generation (1), training data and recipe.** Values from the experiment log, measured on the retrieval output of the time.

| Variant | Description | BS F1 | Tok F1 | Score |
|---|---|---|---|---|
| v1 | Free-form 1,500 examples, 4 attention modules | 0.7776 | 0.3788 | 30.90 |
| v2 | Forced short template (54 chars avg.), 5 epochs, r 32 | 0.6618 | 0.1941 | 23.73 |
| v3 | Templated opening + free explanation (110 chars avg.) | 0.6752 | 0.2085 | 24.43 |
| CPT | 4,231 raw chunks on top of v1 | 0.7752 | 0.3719 | 30.69 |
| Mixed | 74% raw chunks + 26% QA, from base | 0.7605 | 0.3546 | 29.91 |
| Seven-module | v1 data, MLP added | 0.7807 | 0.3852 | 31.12 |
| Hard-example | + 400 QA on weak violation types | 0.7796 | 0.3811 | 31.01 |
| v6_a / v6_b / v6_f | 5,000 v6 examples, trained from base | 0.7761 / 0.7752 / 0.7792 | 0.3820 / 0.3819 / 0.3873 | 30.92 / 30.90 / 31.12 |

On top of v1, three few-shot examples gave 31.05 (+0.15); eight context chunks (30.90), truncation to two sentences (30.79), and max_new_tokens 150 (31.05) gave no gain.

**Generation (2), final retrieval output held fixed.**

| Configuration | BS F1 | Tok F1 | Score | s/question |
|---|---|---|---|---|
| Seven-module v1 adapter | 0.7827 | 0.3979 | 31.44 | 4.75 |
| v6_c (continued from v1) | 0.7830 | 0.4000 | 31.49 | 5.66 |
| v6_c + 3-shot + prompt v2 | 0.7734 | 0.3885 | 30.97 | 6.24 |
| v6_c + 3-shot + strong post-processing | 0.7801 | 0.3936 | 31.28 | 6.21 |
| v6_c + 3-shot + temperature 0.3 | 0.7796 | 0.3910 | 31.21 | 6.23 |
| v6_c + all of the above | 0.7229 | 0.2968 | 27.62 | 6.63 |

The log credits v6_c with +0.37 (31.12 to 31.49), but 31.12 was measured on older retrieval output. With retrieval held fixed, the gain is +0.05. Variant v6_g, which set dropout 0.1, matched v6_c to the last digit because the continued-training path inherits the starting adapter's configuration.

## 5. Limitations and future work

- **Single eval set.** Every selection was made on the same 278 questions. The pool-6 R@5 spread across runs is 1.19 points, larger than the measured effects of normalization, dual RRF, and v6_c. A separate development and final-check split, plus repeated runs, would be needed to confirm those choices.
- **Chunk-level separation is incomplete.** Of the 566 eval gold chunks, 70 were seed chunks for v1 training QA and 76 appear in v6 answer-chunk lists. The v6 generator filters seeds only, so chunks added by the adjacent and scattered modes are not checked.
- **BM25 weight not propagated.** Runs pass `--bm25_weight 0.3`, but the HybridRetriever used for bgem3ko and arctic does not receive it when loaded into the ensemble and falls back to `DEFAULT_ALPHA = 0.2`. All reported numbers reflect that state.
- **No end-to-end latency check.** The FastAPI router in the repo is a stub, and latency was measured per stage (rerankers 1.71 s and 1.88 s, generation 5.66 s on an RTX 5070 Ti). The 30-second limit still needs an end-to-end test on the serving path.
- **Open items.** The v6_c weights are not in the repo and must be retrained from the saved command. DPO, multi-candidate generation with selection, and sentence-level context compression were not tried.

## References

- Code: https://github.com/tjehdgus/KFTC
- Development log (Tistory, Korean): https://mynews76154.tistory.com/1
- Models: EXAONE 3.5 7.8B Instruct, dragonkue/BGE-m3-ko, dragonkue/snowflake-arctic-embed-l-v2.0-ko, mxbai-rerank-large-v2, bge-reranker-v2-m3, klue/roberta-base
- Methods: BM25, Reciprocal Rank Fusion, LoRA, QLoRA (NF4), Anthropic Contextual Retrieval
