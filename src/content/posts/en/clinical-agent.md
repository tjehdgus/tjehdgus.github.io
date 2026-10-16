---
title: "Clinical Trial Termination Risk Agent: Parallel LangGraph Sub-agents, Dual-Model What-if Re-prediction, and a Data-grounded Verification Loop"
description: "A LangGraph multi-agent system that predicts the termination probability of a trial design, re-predicts proposed revisions with two models before judging them, and reviews protocol drafts against MFDS and ICH guideline text via RAG. Citations in the generated report are verified against data rather than by another LLM."
date: 2026-10-16
lang: en
key: clinical-agent
tags: [LangGraph, Multi-Agent, XGBoost, RAG, Clinical Trials, Hallucination Detection]
tistory: https://mynews76154.tistory.com/4
---

## Abstract

This post describes a clinical trial early-warning agent built for Track 3 (Regulatory Affairs and Intelligent Trial Design) of the 4th AI Drug Discovery Competition in Korea (July to August 2026). Given a drug, indication, phase and an optional protocol draft, the agent returns a termination probability, similar past failures, safety and domestic regulatory context, a judged re-prediction of proposed design revisions, and a regulatory checklist review. A LangGraph graph runs five sub-agents in parallel and passes the synthesized report through a what-if node and a verification node. The termination model (XGBoost, 12,717 ClinicalRisk trials) reached AUC 0.730 on a random split, AUC 0.757 on a temporal split, Brier 0.208 and ECE 0.018. All 53 NCT citations in seven reports existed in the data, and 32 of 38 citations that asserted a reason (84.2%) matched the actual stop reason. Because 97.9% of the full model's feature importance sat in text features, the model ignored design changes; this was handled with a separate design-only model (AUC 0.664) and a rule that requires both models to agree on direction.

## 1. Problem

Classifying `why_stopped` for 3,078 terminated trials from ClinicalTrials.gov with keyword rules gives: unclassified 44.0%, patient recruitment failure 34.3%, business or funding 12.9%, lack of efficacy 6.0%, safety 2.8%. Recruitment failure is the largest single reason, and it depends on factors such as eligibility criteria and enrollment plans that can be changed before a protocol is finalized. The task was framed as an agent that warns, with evidence, about preventable failure risk at design time.

- Input: `drug`, `condition`, `phase`, and optionally a protocol document (PDF/DOCX/TXT) or five design fields (`criteria`, `min_age`, `max_age`, `sex`, `sponsor_type`)
- Output: termination probability, supporting NCT cases, openFDA signals, domestic trial landscape, enrollment-duration and delay-cost simulation, a re-prediction verdict (`reduced` / `increased` / `uncertain` / `noise`), and a 15-item regulatory compliance score

```text
termination prediction   AUC, Accuracy, Brier, ECE
stop-reason classifier   Accuracy, Macro-F1 (300 expert-labeled trials)
embedding retrieval      P@K, MRR@10 (relevant = same expert label as the query)
report faithfulness      faithfulness = existing NCTs / cited NCTs
claim consistency        rate = consistent / asserted   (citations without a stated reason excluded)
regulatory review        compliance = (present + 0.5*weak) / 15 * 100
```

Three constraints applied. Prediction inputs had to be limited to information available at design time. Generated reports have no reference text, so faithfulness was measured by checking against data rather than with reference-based metrics. The preliminary round was a proposal review with a 10-page limit. The entry was submitted under a team name, but every commit in the repository is from my account, so this post describes the system rather than splitting contributions.

## 2. Data

| Source | Size | Use |
|---|---|---|
| ClinicalTrials.gov, terminated trials (global) | 3,078 | failure patterns, fact-check reference |
| ClinicalTrials.gov, trials run in Korea | 2,000 | domestic landscape |
| MFDS trial approvals (data.go.kr) | 3,000 | domestic regulatory context |
| ClinicalRisk (Zenodo 7982426) | 12,717 | termination model training and evaluation |
| Single-label failed trials from the above | 5,391 | gold set for classification and retrieval |
| Failed trials with stop-reason text from the above | 5,696 | failure-case RAG corpus |
| MFDS and ICH guideline PDFs | 29 documents (7 ICH, 22 MFDS), 1,783 chunks | regulatory RAG source text |
| Curated regulatory requirement KB | 43 entries | review criteria |

- Labels: `Completed = 0`; Terminated/Withdrawn/Suspended `= 1`. Outcome fields such as `why_stopped` were excluded from the inputs.
- Features: TF-IDF (8,000 dims) over eight text fields (title, condition, intervention, criteria, study design, outcome, sponsors, location) plus 21 structured features (min/max age and span, criteria length and line count, one-hot sex, study type, healthy volunteers and sponsor type, among others).
- Guideline chunking: text was extracted with pdfplumber, page numbers, dotted TOC lines and headers were stripped, and documents were split on clause headings (`제N조`, `5.2.1`, `ANNEX`), then re-split on sentence boundaries when too long (`MIN_CHARS=200`, `MAX_CHARS=1200`, `OVERLAP=150`). Each chunk keeps document title, clause and URL. The PDFs are copyrighted, so only `manifest.json` was committed.

## 3. Method

### 3.1 Pipeline

```text
input (drug, indication, phase, protocol draft)
  → coordinator      extract five design fields from the draft (forced tool call)
  ├─ predict         XGBoost termination probability + per-condition failure stats
  ├─ research        failure-case RAG (BGE-m3-ko, 5,696 trials)
  ├─ safety          openFDA adverse event signals
  ├─ regulatory      domestic landscape + protocol review (regulatory RAG)
  └─ feasibility     Monte Carlo enrollment duration and delay cost
  → synthesize       merge five opinions into one report
  → whatif           revisions → design fields → re-predict with two models → verdict
  → verify           NCT existence check + claim consistency check + LLM sufficiency review
       ├─ pass, or 2 iterations → END (warning prepended if unresolved)
       └─ fail → coordinator (feedback injected, re-run)
```

- coordinator: extracts model input fields from the draft first; without this, the pre-revision risk would be computed from defaults.
- sub-agents: numbers come from models, retrieval, APIs and the simulator; the LLM only interprets them into a 3-4 sentence opinion.
- whatif / verify: revision effects are checked by re-prediction, report evidence by data lookup.

All LLM calls go through a single adapter, `llm.py`, which converts Anthropic-style tool schemas (`input_schema`) into OpenAI function specs; the runtime default is `gpt-4o-mini`. The web app is FastAPI with SSE streaming of node events.

### 3.2 LangGraph graph

State is a `TypedDict`, and events from parallel nodes accumulate through `Annotated[list, operator.add]`. LangGraph drops returned keys that are not declared in State, so verdict fields such as `whatif_note` and `whatif_verdict` were declared explicitly.

`mvp/agent_graph.py`
```python
    g.add_edge(START, "coordinator")
    # 코어 → 5개 서브에이전트 병렬 위임
    for n in ["predict", "research", "safety", "regulatory", "feasibility"]:
        g.add_edge("coordinator", n)
        g.add_edge(n, "synthesize")   # 5개 완료 후 종합 (LangGraph가 합류 대기)
    g.add_edge("synthesize", "whatif")   # 종합된 수정 제안을 재예측으로 검증
    g.add_edge("whatif", "verify")
    g.add_conditional_edges("verify", _after_verify, {"coordinator": "coordinator", END: END})
```

`_after_verify` returns `END` when `verify_ok` is set or `iterations >= 2`, and `coordinator` otherwise. If verification still fails after the last iteration, `run()` prepends a warning listing the number of unverified citations and reason mismatches and returns `unresolved=True`.

### 3.3 Termination model

The full model is XGBoost (`n_estimators=1200`, `max_depth=5`, `learning_rate=0.03`, `subsample=0.85`, `colsample_bytree=0.6`, `reg_lambda=2.0`, `reg_alpha=0.5`), with 15% of the training set held out for `early_stopping_rounds=40`.

TF-IDF features carried 97.9% of the importance and all structured features together 2.1%. With this model, relaxing the upper age limit or removing a sex restriction changed the predicted risk by 0.0 percentage points, so the model did not respond to the design levers a researcher can actually change. A second model was therefore trained on the 21 structured features only, by slicing the 8,000 text columns off the same feature matrix.

`mvp/train_predictor.py`
```python
    ntext = 8000
    Dtr, Dte = Xtr[:, ntext:], Xte[:, ntext:]
    m = xgb.XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=2.0, eval_metric="auc", n_jobs=4, random_state=42)
    m.fit(Dtr, ytr)
```

### 3.4 What-if re-prediction and verdict

The what-if agent translates the report's natural-language suggestions into the five model fields through a forced tool call (`revise_design`). The prompt forbids changes the report did not propose and relaxations that compromise scientific validity, such as dropping safety exclusions. Only changes that actually alter a model field are reported as `revisions`; narrative additions go to `narrative_revisions`. Both models re-predict the before and after risk, and the verdict is based on whether they agree in direction rather than on whichever looks better.

`mvp/agent_graph.py`
```python
    NOISE = 1.0          # %p — 이 아래는 측정 오차로 본다
    agree = (delta > NOISE and d_delta > NOISE) or (delta < -NOISE and d_delta < -NOISE)
    if abs(delta) <= NOISE and abs(d_delta) <= NOISE:
        note = ("제안된 수정의 위험 변화는 두 모델 모두에서 측정 오차 수준입니다."
                " 주된 위험은 설계값이 아닌 다른 요인에 있습니다.")
    elif not agree:
        note = (f"두 모델의 방향이 엇갈립니다 (전체 {delta:+}%p / 설계 {d_delta:+}%p)."
                f" 수정 효과가 불확실하므로 수치를 근거로 채택하지 마세요.")
```

(The messages say: both changes are within noise; the two models disagree in direction, so do not adopt the revision on the numbers.)

The same node simulates the change in enrollment duration and delay cost. Per-site monthly enrollment is drawn from `Gamma(shape=2, mean=1.2)`, and the screen-failure rate comes from a restrictiveness score (0.30 to 0.55) built from the number of exclusions, text length and numeric thresholds in the criteria, plus noise; 5,000 runs give the median and 90% interval. Since the priors come from industry aggregates, the output states that results are for relative comparison between designs.

### 3.5 Retrieval

Both RAG indexes use normalized `dragonkue/BGE-m3-ko` embeddings and a numpy dot product. The regulatory corpus has 1,826 entries: 43 KB items and 1,783 guideline chunks, with `[source] document · clause` prepended to each chunk before embedding. For each of 15 checklist items combining SPIRIT 2013, ICH-GCP and MFDS requirements, the reviewer injects one criterion (`kind="kb"`) and one guideline excerpt (`kind="doc"`), and the LLM returns `present` / `weak` / `missing` as structured output. Incomplete outputs are retried up to three times and then raise an error instead of silently scoring zero. When generating revision text, the cited clause is overwritten with the item's verified standard instead of trusting the LLM.

### 3.6 Verification node

1. Identifier check: NCT numbers matched by `NCT\d{8}` are looked up in the local datasets (no LLM).
2. Claim consistency: sentences mentioning an NCT are classified with the rule classifier and compared with the trial's actual `failure_category` (no LLM). Citations where the report states no reason (`OTHER`) are counted as `abstained` and excluded from the denominator.
3. Sufficiency review: an LLM checks for a risk figure, evidence, revisions, and safety and domestic context.

On failure, fabricated NCTs, mismatched citations and the reviewer's notes are combined into `fact_feedback` and injected into the next synthesize prompt.

## 4. Experiments

### 4.1 Setup

- Termination model: `train_test_split(test_size=0.2, random_state=42, stratify=y)`, 2,544 test trials, majority-class baseline 0.552.
- Temporal split: trials sorted by NCT number; first 80% (10,173) for training, last 20% (2,544) for evaluation; baseline 0.565.
- Calibration: Brier score and ECE from a 10-bin quantile calibration curve on the same test split, computed on the same sparse matrix used in training and serving. No recalibration (Platt, isotonic) was applied.
- Classification: keyword rules vs. an LLM (`claude-haiku-4-5`, as configured in `evaluate.py` at the time; cost USD 0.301) on 300 gold-set trials.
- Embeddings: 5,391-trial corpus, 300 random queries, self-matches excluded.
- Regulatory reviewer: seven key items were removed one at a time from a complete reference protocol to test detection; the LLM reviewer was also compared with an independent rule-based reviewer on six protocols (90 item comparisons).
- E2E: one full graph run on `gpt-4o-mini` with invariant checks.
- Hardware is not recorded in the repository. All metrics are logged in `output/metrics_evidence.json` with split, seed and model hash.

### 4.2 Results

| Task | Metric | Result |
|---|---|---|
| Termination (random split) | AUC / Accuracy | 0.730 / 66.9% |
| Termination (temporal split) | AUC | 0.757 |
| Calibration | Brier / ECE | 0.208 / 0.0184 |
| Design-only model | AUC | 0.664 |
| Stop-reason classification (LLM) | Accuracy / Macro-F1 | 87.3% / 0.835 |
| Retrieval (BGE-m3-ko) | P@5 / MRR@10 | 0.867 / 0.916 |
| NCT existence | 7 reports | 53/53 |
| Claim consistency | asserted 38, abstained 4 | 32/38 (84.2%), 6 contradictions |
| Regulatory defect detection | 7 injected defects | recall 1.0, 0 false flags, complete protocol scored 90.0 |
| Reviewer convergent validity | missing agreement / kappa / rank corr. | 0.80 / 0.57 / 0.829 |
| E2E run | 4 / 5 sub-agents | 95.8 s / 84.8 s |

The calibration curve was close to the diagonal (predicted 0.16 → observed 0.15, 0.52 → 0.52, 0.78 → 0.78). A deliberately incomplete single-arm gefitinib protocol scored 23.3 (0 present, 7 weak, 8 missing).

### 4.3 Ablation

| Experiment | Variant | Result |
|---|---|---|
| Input ablation (AUC) | majority-class baseline | 0.552 |
| | design features only (21) | 0.668 |
| | text only | 0.716 |
| | full | 0.728 |
| What-if sensitivity (full model) | relax upper age limit | 0.0 pp |
| | remove sex restriction | 0.0 pp |
| Stop-reason classification | keyword rules | 84.0% / F1 0.785 |
| | LLM | 87.3% / F1 0.835 |
| Embeddings (P@5 / MRR) | MiniLM-L12 (previous) | 0.833 / 0.901 |
| | BGE-M3 | 0.849 / 0.919 |
| | KURE-v1 | 0.860 / 0.929 |
| | BGE-m3-ko (selected) | 0.867 / 0.916 |
| Claim-consistency denominator | including citations without a stated reason | 76.2% |
| | asserted citations only | 84.2% |
| Pre-revision risk baseline | default inputs | 0.271 |
| | fields extracted from the protocol | 0.259 |

The ablation values are the `ablation_auc` record in `metrics_evidence.json`; the ablation script itself was removed in a cleanup commit. Design features alone beat the baseline by 0.116 but are masked by text in the full model, which is why the design-only model was added. Retrieval was selected on P@5 because the report shows five cases (six models were compared; KURE-v1 had higher MRR).

## 5. Limitations and future work

- External validity: a blinded expert scoring harness (`expert_eval.py`, six protocols and a scoring sheet) was built but never run. Validation so far covers internal consistency, defect injection and agreement between two independent reviewers.
- The text input includes `sponsors` and `location`, so the model may have learned specific sponsors or sites rather than design. The top TF-IDF tokens should be inspected.
- The temporal split uses NCT number as a proxy for date, and the evaluation window has a different base rate (0.565), so AUC 0.757 should not be read directly as forward generalization.
- Faithfulness 1.0 only means the cited NCT identifiers exist. Claim consistency relies on the rule classifier and does not cover other claims such as numbers or doses.
- The feasibility simulator's priors are assumptions from industry aggregates without empirical calibration.

## References

- Development log (Tistory, Korean): https://mynews76154.tistory.com/4
- Code repository: private
- Data: ClinicalTrials.gov API, openFDA, MFDS clinical trial information (data.go.kr), ClinicalRisk Dataset (Zenodo 7982426), ICH guidelines, MFDS guidance documents
- Models and tools: XGBoost, LangGraph, dragonkue/BGE-m3-ko, KURE-v1, BGE-M3, multilingual-E5-large-instruct, gpt-4o-mini, claude-haiku-4-5
- Standards: SPIRIT 2013, ICH E6(R3), E6(R2), E8(R1), E9, E9(R1), E2A, E3
