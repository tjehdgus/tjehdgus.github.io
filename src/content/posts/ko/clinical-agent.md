---
title: "임상시험 중단 위험 예측 에이전트: LangGraph 병렬 서브에이전트, 이중 모델 what-if 재예측, 데이터 대조 검증 루프"
description: "임상 설계 정보와 계획서 초안으로 중단확률을 예측하고, 수정안을 두 모델로 재예측해 판정하며, 식약처·ICH 원문 RAG로 계획서를 검토하는 LangGraph 멀티에이전트를 정리했다. 리포트의 인용 근거는 LLM이 아닌 데이터 대조로 검증한다."
date: 2026-10-16
lang: ko
key: clinical-agent
tags: [LangGraph, Multi-Agent, XGBoost, RAG, Clinical Trials, Hallucination Detection]
tistory: https://mynews76154.tistory.com/4
---

## 초록

「제4회 AI 신약개발 경진대회」 분야 3(규제 대응 및 지능형 임상 설계)에 제출한 임상 실패 조기경보 에이전트를 정리한다(2026.07~08). 약물·적응증·단계와 계획서 초안을 받아 중단확률, 유사 실패 선례, 안전성·국내 규제 맥락, 수정안 재예측 판정, 규제 체크리스트 검토 결과를 낸다. LangGraph 그래프가 5개 서브에이전트를 병렬 실행하고, 종합 리포트를 what-if 노드와 검증 노드에 통과시킨다. 중단확률 모델(XGBoost, ClinicalRisk 12,717건)은 무작위 분할 AUC 0.730, 시간 분할 AUC 0.757, Brier 0.208, ECE 0.018이었다. 리포트 7건의 인용 NCT 53건이 모두 실재했고, 사유를 특정한 인용 38건 중 32건(84.2%)이 실제 중단 사유와 일치했다. 전체 모델 중요도의 97.9%가 텍스트라 설계 변경에 반응하지 않는 문제는 설계 요인 전용 모델(AUC 0.664)과 두 모델 방향 일치 판정으로 처리했다.

## 1. 문제 정의

ClinicalTrials.gov 중단 임상 3,078건의 `why_stopped`를 키워드 규칙으로 분류하면 미분류 44.0%, 환자 모집 실패 34.3%, 사업·자금 12.9%, 효능 부족 6.0%, 안전성 2.8%다. 단일 사유로는 모집 실패가 가장 많고, 적격기준·모집 계획처럼 설계 단계에서 손볼 수 있는 요인이다. 과제를 "설계 단계에서 예방 가능한 실패 위험을 근거와 함께 경고하는 에이전트"로 정했다.

- 입력: `drug`, `condition`, `phase`, 선택적으로 계획서 원문(PDF/DOCX/TXT) 또는 설계 필드 5종(`criteria`, `min_age`, `max_age`, `sex`, `sponsor_type`)
- 출력: 중단확률, 근거 NCT 사례, openFDA 시그널, 국내 현황, 모집기간·지연비용 시뮬레이션, 재예측 판정(`reduced` / `increased` / `uncertain` / `noise`), 15개 항목 규제 준수 점수

```text
중단확률 예측    AUC, Accuracy, Brier, ECE
실패 사유 분류   Accuracy, Macro-F1 (전문가 라벨 300건)
임베딩 검색      P@K, MRR@10 (검색 결과 라벨 = 질의 라벨이면 relevant)
리포트 사실성    faithfulness = 실재 NCT 수 / 인용 NCT 수
인용 서술 일치   rate = consistent / asserted   (사유 미서술 인용은 분모 제외)
규제 검토        compliance = (present + 0.5*weak) / 15 * 100
```

제약은 세 가지다. 예측 입력은 설계 시점 정보만 쓴다. 생성 리포트는 정답 텍스트가 없어 참조 기반 지표 대신 데이터 대조로 사실성을 잰다. 예선은 본문 10쪽 제안서 평가였다. 팀명으로 접수했으나 레포 커밋은 모두 본인 계정이며, 이 글은 시스템 기준으로 쓴다.

## 2. 데이터

| 소스 | 규모 | 용도 |
|---|---|---|
| ClinicalTrials.gov 글로벌 중단 임상 | 3,078건 | 실패 패턴, 사실 검증 대조 |
| ClinicalTrials.gov 한국 수행 임상 | 2,000건 | 국내 현황 |
| 식약처 임상승인(data.go.kr) | 3,000건 | 국내 규제 맥락 |
| ClinicalRisk (Zenodo 7982426) | 12,717건 | 중단확률 학습·평가 |
| 위 데이터의 단일 라벨 실패 사례 | 5,391건 | 분류·임베딩 정답셋 |
| 위 데이터의 사유 텍스트 있는 실패 사례 | 5,696건 | 실패 사례 RAG 코퍼스 |
| 식약처·ICH 가이드라인 원문 | 29개 문서(ICH 7, 식약처 22), 1,783청크 | 규제 RAG 원문 |
| 규제 요건 큐레이션 KB | 43항목 | 규제 판정 기준 |

- 라벨: `Completed = 0`, Terminated/Withdrawn/Suspended `= 1`. `why_stopped` 등 결과 정보는 입력에서 뺐다.
- 특징: 텍스트 8개 필드(제목·질환·중재·적격기준·설계·평가변수·스폰서·기관)의 TF-IDF 8,000차원과 구조화 특징 21개(연령 하한·상한·범위, 적격기준 길이·줄 수, 성별·연구유형·건강인·스폰서 유형 원핫 등).
- 규제 원문 청킹: pdfplumber 추출 후 페이지 번호·목차 점선·머리말을 지우고, `제N조`, `5.2.1`, `ANNEX` 같은 조항 헤딩으로 나눈 뒤 길면 문장 경계로 재분할한다(`MIN_CHARS=200`, `MAX_CHARS=1200`, `OVERLAP=150`). 청크마다 문서명·조항·URL을 보관한다. 원문은 저작물이라 레포에는 `manifest.json`만 커밋했다.

## 3. 방법

### 3.1 전체 파이프라인

```text
입력(약물·적응증·단계, 계획서 초안)
  → coordinator      계획서에서 설계 필드 5종 추출(강제 도구 호출)
  ├─ predict         XGBoost 중단확률 + 질환별 실패 통계
  ├─ research        실패 사례 RAG(BGE-m3-ko, 5,696건)
  ├─ safety          openFDA 이상사례 시그널
  ├─ regulatory      국내 현황 + 계획서 규제 검토(규제 RAG)
  └─ feasibility     몬테카를로 모집기간·지연비용 시뮬레이션
  → synthesize       5개 소견 종합 리포트
  → whatif           수정안 → 설계 필드 번역 → 두 모델 재예측 → 판정
  → verify           NCT 실재 대조 + 인용 서술 대조 + LLM 충분성 심사
       ├─ 통과 또는 2회 반복 → END (미해결이면 리포트 상단 경고)
       └─ 실패 → coordinator (피드백 주입 후 재조사)
```

- coordinator: 계획서에서 모델 입력 필드를 먼저 추출한다. 없으면 수정 전 위험이 기본값으로 계산된다.
- 서브에이전트: 수치는 모델·검색·API·시뮬레이터가 만들고 LLM은 3~4문장 소견으로 해석만 한다.
- whatif / verify: 수정안 효과는 재예측으로, 리포트 근거는 데이터로 확인한다.

LLM 호출은 `llm.py` 어댑터 하나로 모았고(Anthropic 형식 도구 스키마를 OpenAI function으로 변환, 기본 `gpt-4o-mini`), 웹 앱은 FastAPI + SSE다.

### 3.2 LangGraph 그래프

State는 `TypedDict`이고 병렬 노드 이벤트는 `Annotated[list, operator.add]`로 누적한다. LangGraph는 State에 선언되지 않은 반환 키를 버리므로 `whatif_note`, `whatif_verdict` 같은 판정 필드도 모두 선언했다.

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

`_after_verify`는 `verify_ok`이거나 `iterations >= 2`이면 `END`, 아니면 `coordinator`를 반환한다. 반복 상한 후에도 검증이 통과하지 못하면 `run()`이 리포트 맨 위에 미확인 인용 수와 사유 불일치 수를 적은 경고를 붙이고 `unresolved=True`를 반환한다.

### 3.3 중단확률 모델

전체 모델은 XGBoost(`n_estimators=1200`, `max_depth=5`, `learning_rate=0.03`, `subsample=0.85`, `colsample_bytree=0.6`, `reg_lambda=2.0`, `reg_alpha=0.5`)이며 학습셋 15%를 검증셋으로 떼어 `early_stopping_rounds=40`을 걸었다.

특징 중요도의 97.9%가 TF-IDF, 구조화 특징 전체가 2.1%였다. 이 모델로 연령 상한을 완화하거나 성별 제한을 풀면 위험 변화가 0.0%p였다. 연구자가 바꿀 수 있는 설계값에 반응하지 않는 모델이다. 그래서 같은 특징 행렬에서 텍스트 8,000열을 잘라낸 구조화 21개 특징만으로 설계 요인 전용 모델을 따로 학습했다.

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

### 3.4 what-if 재예측과 판정

whatif 에이전트가 리포트의 자연어 수정 제안을 강제 도구 호출(`revise_design`)로 5개 모델 필드로 번역한다. 리포트에 없는 수정, 안전성 배제기준 삭제 같은 완화는 금지했다. 모델 필드 값이 실제로 바뀐 변경만 `revisions`로, 서술 보강은 `narrative_revisions`로 분리한다. 두 모델로 전후 위험을 계산하고 유리한 쪽을 고르지 않고 방향 일치로 판정한다.

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

같은 노드에서 모집기간·지연비용 변화도 시뮬레이션한다. 기관별 월 등록률은 `Gamma(shape=2, mean=1.2)`, 스크리닝 실패율은 적격기준의 배제 조건 수·길이·수치 개수로 만든 엄격도(0.30~0.55)에 잡음을 더해 뽑고 5,000회 반복한다. 사전분포가 업계 집계 기반이라 출력에 "설계 간 상대 비교로 해석"을 명시했다.

### 3.5 근거 검색

두 RAG 모두 `dragonkue/BGE-m3-ko` 정규화 임베딩과 numpy 내적 검색이다. 규제 코퍼스는 KB 43항목과 원문 1,783청크를 합친 1,826항목이고, 원문은 `[출처] 문서명 · 조항`을 앞에 붙여 임베딩한다. 규제 검토는 SPIRIT 2013·ICH-GCP·식약처 15개 항목마다 판정 기준(`kind="kb"`) 1건과 원문 발췌(`kind="doc"`) 1건을 프롬프트에 넣고 LLM이 `present` / `weak` / `missing`을 구조화 출력으로 낸다. 항목이 덜 오면 3회까지 재시도하고 그래도 부족하면 예외를 낸다. 수정 문안 생성은 근거 조항을 LLM 출력 대신 해당 항목의 표준으로 덮어쓴다.

### 3.6 검증 노드

1. 식별자 대조: `NCT\d{8}`로 뽑은 번호를 보유 데이터와 집합 대조한다(LLM 미사용).
2. 서술 대조: NCT가 언급된 문장을 규칙 분류기로 분류해 실제 `failure_category`와 비교한다(LLM 미사용).
3. 충분성 심사: 위험 수치, 근거, 수정안, 안전성·국내 맥락을 LLM이 본다.

서술 대조에서 리포트가 사유를 특정하지 않은 인용(규칙 분류 결과 `OTHER`)은 `abstained`로 분리해 분모에서 뺀다. 실패하면 지어낸 NCT, 사유가 어긋난 인용, LLM 지적 사항을 `fact_feedback`으로 묶어 다음 synthesize 프롬프트에 넣는다.

## 4. 실험

### 4.1 실험 설정

- 중단확률: `train_test_split(test_size=0.2, random_state=42, stratify=y)`, 테스트 2,544건, 다수 클래스 기저 0.552.
- 시간 분할: NCT 번호 순 앞 80%(10,173건) 학습, 뒤 20%(2,544건) 평가, 기저 0.565.
- 보정: 같은 테스트 분할에서 Brier와 10분위 보정곡선 ECE. 학습·운영과 같은 희소 행렬로 계산했다. 재보정(Platt, isotonic)은 적용하지 않았다.
- 분류: 정답셋 300건에서 키워드 규칙과 LLM(당시 `evaluate.py` 설정 `claude-haiku-4-5`, 비용 0.301달러) 비교.
- 임베딩: 코퍼스 5,391건, 무작위 질의 300건, 자기 자신 제외.
- 규제 검토기: 15개 항목을 갖춘 기준 계획서에서 핵심 7개 항목을 하나씩 지워 탐지 여부를 보고, 규칙 기반 검토기와 LLM 검토기를 계획서 6건(항목 비교 90건)에서 대조했다.
- E2E: `gpt-4o-mini`로 전체 그래프 완주와 불변식 확인.
- 하드웨어 사양은 레포에 기록이 없다. 수치는 `output/metrics_evidence.json`에 분할·시드·모델 해시와 함께 남겼다.

### 4.2 결과

| 작업 | 지표 | 결과 |
|---|---|---|
| 중단확률(무작위 분할) | AUC / Accuracy | 0.730 / 66.9% |
| 중단확률(시간 분할) | AUC | 0.757 |
| 확률 보정 | Brier / ECE | 0.208 / 0.0184 |
| 설계 요인 전용 모델 | AUC | 0.664 |
| 실패 사유 분류(LLM) | Accuracy / Macro-F1 | 87.3% / 0.835 |
| 임베딩 검색(BGE-m3-ko) | P@5 / MRR@10 | 0.867 / 0.916 |
| 인용 NCT 실재율 | 리포트 7건 | 53/53 |
| 인용 서술 일치 | asserted 38, abstained 4 | 32/38 (84.2%), 모순 6 |
| 규제 결함 탐지 | 심은 결함 7개 | recall 1.0, 오탐 0, 충실판 90.0점 |
| 검토기 수렴 타당도 | 결여 일치 / κ / 순위 상관 | 0.80 / 0.57 / 0.829 |
| E2E 완주 | 서브에이전트 4개 / 5개 | 95.8초 / 84.8초 |

보정곡선은 예측 0.16 → 실제 0.15, 0.52 → 0.52, 0.78 → 0.78이었다. 허점을 둔 게피티닙 단일군 계획서 샘플은 23.3점(충족 0, 미흡 7, 누락 8)이었다.

### 4.3 비교 실험

| 실험 | 변형 | 결과 |
|---|---|---|
| 입력 절제(AUC) | 다수 클래스 기저 | 0.552 |
| | 설계 특징 21개만 | 0.668 |
| | 텍스트만 | 0.716 |
| | 전체 | 0.728 |
| what-if 민감도(전체 모델) | 연령 상한 완화 | 0.0%p |
| | 성별 제한 해제 | 0.0%p |
| 실패 사유 분류 | 키워드 규칙 | 84.0% / F1 0.785 |
| | LLM | 87.3% / F1 0.835 |
| 임베딩(P@5 / MRR) | MiniLM-L12 (기존) | 0.833 / 0.901 |
| | BGE-M3 | 0.849 / 0.919 |
| | KURE-v1 | 0.860 / 0.929 |
| | BGE-m3-ko (채택) | 0.867 / 0.916 |
| 인용 서술 일치 분모 | 사유 미서술 인용 포함 | 76.2% |
| | 사유를 특정한 인용만 | 84.2% |
| 수정 전 위험 기준선 | 기본값 입력 | 0.271 |
| | 계획서 추출 필드 | 0.259 |

절제 값은 `metrics_evidence.json`의 `ablation_auc` 기록이다(절제 스크립트는 정리 커밋에서 삭제). 설계 특징은 단독으로 기저보다 0.116 높지만 전체 모델에서는 텍스트에 가려지며, 이것이 설계 전용 모델을 둔 근거다. 임베딩은 리포트가 사례 5건을 보여주므로 P@5로 골랐다(6종 비교, MRR은 KURE-v1이 높음).

## 5. 한계와 향후 과제

- 외적 타당성: 전문가 블라인드 채점 하네스(`expert_eval.py`, 계획서 6건과 채점표)는 만들었지만 실행하지 못했다. 현재 검증은 내부 일관성, 결함 주입, 두 독립 검토기의 수렴까지다.
- 텍스트 입력에 `sponsors`, `location`이 있어 모델이 설계 대신 특정 스폰서·기관을 학습했을 수 있다. 상위 토큰 분석이 필요하다.
- 시간 분할은 NCT 번호를 날짜 대리변수로 썼고 평가 구간 기저율(0.565)이 달라 AUC 0.757을 그대로 일반화 성능으로 읽기 어렵다.
- 사실성 1.0은 NCT 식별자 실재율이다. 서술 대조도 규칙 분류기 기반이라 수치·용량 같은 주장은 검증하지 않는다.
- 경제성 시뮬레이터의 사전분포는 업계 집계 가정이며 실측 보정이 없다.

## 참고

- 티스토리 개발 기록: https://mynews76154.tistory.com/4
- 코드 레포: 비공개
- 데이터: ClinicalTrials.gov API, openFDA, 식약처 의약품 임상시험 정보(data.go.kr), ClinicalRisk Dataset(Zenodo 7982426), ICH 가이드라인, 식약처 민원인안내서
- 모델·도구: XGBoost, LangGraph, dragonkue/BGE-m3-ko, KURE-v1, BGE-M3, multilingual-E5-large-instruct, gpt-4o-mini, claude-haiku-4-5
- 표준: SPIRIT 2013, ICH E6(R3), E6(R2), E8(R1), E9, E9(R1), E2A, E3
