---
title: "Maeum Diary: Golden-Set-Driven Emotion Classification Prompts and a Conversational Agent with pgvector and Knowledge-Graph Memory"
description: "A technical report on the emotion analysis, RAG long-term-memory chat, and crisis detection layer of the AI diary app Maeum Diary. On a 394-item golden set, emotion classification accuracy rose from 53.8% to 71.6% and Macro F1 from 0.483 to 0.700."
date: 2026-08-04
lang: en
key: maeum-diary
tags: [LLM, RAG, pgvector, LangGraph, Prompt Evaluation, FastAPI, AI Safety]
tistory: https://mynews76154.tistory.com/7
---

## Abstract

Maeum Diary is a PWA where an LLM analyzes the emotions in each diary entry and the user can chat with an AI that remembers past entries. It was built between March and July 2026 and is deployed at https://diary-ai-app-chi.vercel.app. Emotion analysis has Claude haiku-4-5 emit JSON with one of eight emotion groups plus a free-form specific emotion. The chat feature, "Bamboo Forest", is a LangGraph agent whose tools are pgvector semantic search, a user-fact memory, and a per-user knowledge graph. The classification prompt was revised twice against an emotion codebook and a 394-item golden set, moving Accuracy from 53.8% to 71.6%, Macro F1 from 0.483 to 0.700, and Cohen's Kappa from 0.469 to 0.670. Crisis signals are detected by OR-combining a regex keyword screen with the LLM's own judgment, and conversations with a crisis signal never enter the agent loop. Apart from a teammate's work in the first two weeks (biorhythm research, synthetic user personas), every system described here is my own.

## 1. Problem

The input is a single diary entry. The system produces four outputs.

| Output | Form |
|---|---|
| Emotion analysis | `primary_group` (one of 8), `specific_emotion` (free-form), `intensity` (1-10), `mixed`, `summary`, `risk_level`, `confidence` |
| Action guide | 2-3 suggested actions, or a rest message when intensity is 7 or higher |
| Chat reply | 3-5 casual, warm sentences in the Bamboo Forest persona |
| Crisis notice | `none / caution / high` with hotline information |

The eight groups (positive, sadness, anger, anxiety, lethargy, loneliness, confusion, mixed) drive charts and statistics. The free-form emotion ("anxious excitement") is what the user sees.

Evaluation metrics (`scripts/evaluate_emotion_analysis.py`):

```text
Accuracy      = share of entries whose predicted group matches the label
                (parse failures and out-of-set predictions count as an 'invalid' label)
Macro F1      = unweighted mean of per-group F1 over the 8 groups
Cohen's Kappa = (p_o - p_e) / (1 - p_e)
Intensity     = MAE, Pearson r
```

The label distribution is skewed (85 positive vs. 25 loneliness), so prompts were accepted on Macro F1 and per-group recall rather than Accuracy.

Constraints: a single part-time developer; free-tier infrastructure (Render sleeps when idle, Voyage free tier is 3 RPM); strict isolation of diaries between users; no diary text sent to external tracing services; and recall over precision for crisis signals.

## 2. Data

No real user data was used for evaluation.

| Dataset | Size | Use |
|---|---|---|
| Seed diaries | 100 synthetic users × 90 days = 9,000 entries | dev DB, RAG search demos |
| Noisy diaries | 1,061 entries (ultra_short 356 / slang_typo 282 / mixed_emotion 213 / factual_no_emotion 210) | first evaluation set, golden-set candidate pool |
| Golden set v1 | 394 entries (slang_typo 129 / factual 124 / ultra_short 95 / mixed 46) | prompt acceptance |

The seed diaries were too clean, so I generated noisy entries of four types: one-liners, broken spelling and slang, emotionless lists of facts, and long entries with conflicting emotions. Their labels came from the generating LLM, and tuning against them would only fit that model's opinion. The golden set was built separately.

```text
noisy 1,061 → stratified sample 400 by group × noise type (seed 42)
  ├─ human review 28
  ├─ judge prompt (Claude + codebook rules) re-labels independently
  │    → agrees with generator label: 236 (llm_consensus)
  └─ disputes 133 → third model (GPT family) votes
       → 2:1 majority: 130 (llm_majority)
       → three-way split: 3, dropped
  → 3 judge failures dropped → golden set 394
```

Labels follow `docs/emotion-codebook.md`. Its core is a set of boundary rules derived from baseline errors.

| Rule | Content |
|---|---|
| R1 | "Mixed" is the last resort. If one emotion is even slightly dominant, pick it |
| R2 | Missing a person or connection → loneliness; something lost → sadness; energy → lethargy |
| R3 | Fact-only entries still get the closest group, at intensity 1-3 |
| R4 | A clear object of fear → anxiety; not knowing itself is the point → confusion |
| R5 | When surface and underlying emotions are balanced, prefer the surface one |
| R6 | If a loss event is central, default to sadness (added in round 3) |

Embeddings use Voyage `voyage-3.5` (1024 dims), asymmetrically: diaries as `document`, queries as `query`. Entries are not chunked; each entry (first 16,000 characters) is one vector.

## 3. Method

### 3.1 Pipeline

```text
[POST /diaries]
  insert diary
  ├─ emotion analysis (Claude haiku-4-5, one retry if JSON parsing fails)
  │    └─ crisis: keyword_screen OR LLM risk_level; log when confidence < 0.6
  ├─ action guide (takes the emotion JSON as input)
  └─ BackgroundTasks: Voyage embedding → knowledge-graph extraction → "what changed" line

[POST /chat]
  last 30 messages + diaries from the last 7 days (sleep, condition, emotion score)
  ├─ crisis guard: keyword_screen != none → single tool-less call + hotline notice
  ├─ 1st: v2 LangGraph  classify → specialist (strategy note, tools) → voice
  ├─ 2nd: v1 single tool-use loop (max 5 iterations)
  └─ 3rd: legacy  code runs pgvector search → inject into prompt → single call
```

Emotion analysis is synchronous because the result is shown right after saving; embedding and graph extraction run in the background so they never block the save. In chat, each path falls through to the next one on an exception. The text LLM switches to GPT-4o-mini only on 401/403 or exhausted credit; 429s and timeouts are treated as transient.

### 3.2 Emotion analysis prompt

Analysis and the empathy message come back from one call as JSON. The prompt's classification-rules section is a direct port of the codebook. The other sections each target an observed failure: a rule against distorting facts (an empathy line turned "I ate and went to bed" into "you went to bed without eating"); a rule to treat every input as a diary entry (inputs like "test" made the model chat back and break the JSON contract); and, for long voice-dictated entries, a rule to take the emotion that lasted to the end as primary and put the ones passed through into `mixed`.

Even if parsing fails twice, the keyword crisis screen still runs on the raw text. The action guide is a second call that takes the emotion JSON as input.

### 3.3 Crisis detection

`services/risk_detection.py` runs 9 high and 9 caution regexes over both the original text and a whitespace-stripped copy (to catch spaced-out variants), then takes the higher of the keyword level and the LLM level.

`backend/app/services/risk_detection.py`

```python
def combine_risk(keyword_level: str, llm_level: str | None) -> str:
    """
    키워드 스크린과 LLM 판정을 결합해 최종 등급 산출.
    보수적 원칙: 두 판정 중 더 높은 등급을 따른다 (OR 결합).
    LLM 판정이 없거나(파싱 실패 등) 무효 값이면 키워드 결과만 사용.
    """
    if llm_level not in RISK_LEVELS:
        if llm_level is not None:
            logger.warning("[위기 감지] LLM risk_level 무효 값: %r → 키워드 결과만 사용", llm_level)
        llm_level = "none"
    kw = keyword_level if keyword_level in RISK_LEVELS else "none"
    return max(kw, llm_level, key=RISK_LEVELS.index)
```

(The docstring says: take the higher of the two levels; fall back to the keyword result if the LLM level is missing or invalid.) On detection the app only shows hotline information (109, 1577-0199, 1388); it takes no automatic action. Patterns, figurative expressions such as "so annoyed I could die" (must be `none`), and the combination rule are covered by regression tests in CI.

### 3.4 RAG long-term memory

There is no separate vector database. `diaries.embedding vector(1024)` carries an HNSW index (`vector_cosine_ops`). Because search narrows by `user_id` in the same table before ordering by distance, user isolation is enforced by a single SQL filter.

`backend/app/services/diary_search.py`

```python
    # cosine_distance: pgvector의 <=> 연산자 (작을수록 유사)
    distance = Diary.embedding.cosine_distance(query_vector)

    rows = (
        db.query(Diary, distance.label("distance"))
        .filter(Diary.user_id == user_id)          # ← 이 한 줄이 "개인별 인덱스" 역할
        .filter(Diary.embedding.isnot(None))       # 임베딩 없는 일기는 검색 불가
        .filter(Diary.created_at < cutoff)
        .order_by(distance)
        .limit(limit)
        .all()
    )
```

Chat retrieval returns the top 3, excludes the last 7 days (already in context), and drops anything beyond cosine distance 0.65, an empirical starting value.

Vector search answers "did something similar happen before?" but not "how has it been going with that person?". So, after each save, an LLM extracts people, events, places, topics, and routines with their relations into `graph_nodes` and `graph_edges`. The extraction prompt includes the existing node list so that aliases such as "gf" and "girlfriend" merge into one node. Facts learned in chat (schedules, relationships) go into `user_facts`, and the latest 20 are injected into the system prompt.

### 3.5 Conversational agent

v1 is a single Claude tool-use loop with five tools (`search_diaries`, `get_emotion_history`, `get_sleep_pattern`, `explore_graph`, `save_user_fact`), at most 5 iterations, and tool results truncated to 4,000 characters.

v2 wires classify, specialist, and voice nodes with LangGraph. The specialists (sinking, surging, tangled, everyday) never speak to the user; they only write a four-line strategy note: real emotion, approach, what to avoid, relevant memory. The persona lives only in the voice node, so a misclassification changes the strategy but not the voice.

`backend/app/services/chat_graph.py`

```python
def _route_after_classify(state: AgentState) -> str:
    # 일상계 + 도구 불필요 → 전문가 생략, 보이스 직행 (비용 절약)
    if state["specialist"] == "일상계" and not state.get("needs_tools"):
        return "voice"
    return "specialist"


_builder = StateGraph(AgentState)
_builder.add_node("classify", classify_node)
_builder.add_node("specialist", specialist_node)
_builder.add_node("voice", voice_node)
_builder.set_entry_point("classify")
_builder.add_conditional_edges("classify", _route_after_classify,
                               {"specialist": "specialist", "voice": "voice"})
_builder.add_edge("specialist", "voice")
_builder.add_edge("voice", END)
GRAPH = _builder.compile()
```

(Everyday small talk that needs no tools skips the specialist to save cost.) The crisis decision is made deterministically by the keyword screen before entering the graph, and on the crisis path the hotline text is appended by code, not generated by the model.

### 3.6 Operations

| Item | Implementation |
|---|---|
| Rate limiting | slowapi keyed on a token hash: 120/min overall, 10/min for LLM endpoints |
| Cost tracking | tokens of every Claude, GPT, and Voyage call logged to `api_usage` by `purpose` |
| Privacy | Sentry with `send_default_pii=False`; LangSmith env vars stripped at startup in production (covered by a test) |
| Schema and CI | Alembic (14 revisions including baseline); GitHub Actions runs 14 pytest tests and a frontend build |
| Scheduling | Actions cron jobs for daily push, hourly reminders, weekly digest, keep-alive |

Features outside the launch scope (saju, DALL-E emotion crystals) are disabled behind a flag.

## 4. Experiments

### 4.1 Setup

- Evaluation set: the 394-item golden set, one run per prompt version. The app's `analyze_emotion()` is called as-is, so retries and post-processing are included.
- Model: `claude-haiku-4-5-20251001`, `max_tokens=2048`, default temperature.
- Versions: baseline (before codebook rules), v2 (R1-R5 and a confidence field), v3 (adds R6).
- All numbers were recomputed from the three result files in the repository and match the commit records.

### 4.2 Results

| Metric | baseline | v2 | v3 |
|---|---|---|---|
| Accuracy | 53.8% | 69.0% | **71.6%** |
| Macro F1 | 0.483 | 0.649 | **0.700** |
| Kappa | 0.469 | 0.637 | **0.670** |
| Invalid predictions | 19 | 8 | 7 |
| Intensity MAE | 0.85 | **0.78** | 0.84 |

| Recall by group (support) | baseline | v2 | v3 |
|---|---|---|---|
| Positive (85) | 0.71 | 0.81 | 0.81 |
| Lethargy (72) | 0.71 | 0.82 | 0.85 |
| Anxiety (69) | 0.46 | 0.65 | 0.55 |
| Anger (43) | 0.40 | 0.56 | 0.47 |
| Confusion (40) | 0.42 | 0.70 | 0.60 |
| Mixed (30) | 0.93 | 0.80 | 0.90 |
| Sadness (30) | 0.10 | 0.17 | 0.80 |
| Loneliness (25) | 0.16 | 0.72 | 0.76 |

Accuracy by noise type, baseline → v3: ultra_short 62.1 → 77.9%, slang_typo 53.5 → 72.1%, factual_no_emotion 47.6 → 66.1%, mixed_emotion 54.3 → 71.7%.

The baseline predicted "mixed" for 130 of 394 entries (30 true), had 0.16 recall on loneliness, and reached only 47.6% on fact-only entries. R1-R3 targeted exactly these, and v2 cut "mixed" predictions to 54. v2 still got only 5 of 30 sadness entries right: after a loss such as failing an exam or an audition, self-blame was read as lethargy, resentment as anger, and worry about the next attempt as anxiety. R6 (a central loss defaults to sadness; tense separates what already happened from what has not) raised sadness recall to 0.80.

### 4.3 Ablation

| Variant | Result | Decision |
|---|---|---|
| Baseline evaluated on the noisy set (1,061 LLM labels) | Acc 68.0%, Kappa 0.608 vs. 53.8% / 0.469 on the golden set | LLM labels inflated the score; acceptance switched to the golden set (68.0% is the value reported in commits) |
| + codebook R1-R5 (v2) | Macro F1 +0.166, loneliness recall 0.16 → 0.72 | Adopted |
| + sadness rule R6 (v3) | Sadness recall 0.17 → 0.80, Macro F1 +0.051 | Adopted |
| Side effects of R6 | Anger recall 0.56 → 0.47, anxiety 0.65 → 0.55, confusion 0.70 → 0.60, MAE 0.78 → 0.84 | Accepted: for a care app, reading a sad user as angry is the worse error. Logged for the next round |
| Keyword screen alone (6 crisis positives in golden set) | 0/6 detected, 2 false cautions among 388 negatives | All positives are fact-only entries with no crisis words, so keywords alone are insufficient. LLM risk output was not stored, so it was not measured |
| Full human review of the golden set | Stopped at 28 items | Switched to hybrid review |

## 5. Limitations and future work

1. Only 7% of the golden set (28/394) is human-reviewed. 71.6% is closer to "agreement with a codebook-guided LLM consensus" than to agreement with people. Sampling and reviewing real entries is still on the roadmap.
2. The same golden set was used both to revise and to evaluate prompts. With no hold-out, v3 may be partly overfit, and with one run per version the run-to-run variance is unknown.
3. Retrieval and agent reply quality were not measured quantitatively. Setting the 0.65 distance threshold properly needs a Recall@k set with gold entries per query.
4. Crisis detection lacks an evaluation set: only six positives, and the LLM's judgment was not recorded.
5. The Alembic baseline is an empty revision, so a fresh database has to be bootstrapped by a separate script, and row-level security is not applied yet.

## References

- Development log (Tistory, Korean): https://mynews76154.tistory.com/7
- Service: https://diary-ai-app-chi.vercel.app
- Models and tools: Claude haiku-4-5, GPT-4o-mini, Voyage voyage-3.5, pgvector (HNSW), LangGraph, FastAPI, React PWA, Supabase PostgreSQL
