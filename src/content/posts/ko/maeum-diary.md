---
title: "마음일기: 골든셋 기반 감정 분류 프롬프트 개선과 pgvector·지식 그래프 기억을 쓰는 대화 에이전트"
description: "AI 일기 앱 「마음일기」의 감정 분석, RAG 장기기억 대화, 위기 감지 레이어를 정리한 기술 리포트. 394건 골든셋 기준으로 감정 분류 정확도를 53.8%에서 71.6%, Macro F1을 0.483에서 0.700으로 올렸다."
date: 2026-08-04
lang: ko
key: maeum-diary
tags: [LLM, RAG, pgvector, LangGraph, Prompt Evaluation, FastAPI, AI Safety]
tistory: https://mynews76154.tistory.com/7
---

## 초록

「마음일기」는 일기를 쓰면 LLM이 감정을 분석하고, 과거 일기를 기억하는 AI와 대화할 수 있는 PWA 서비스다(2026.03~07 개발, https://diary-ai-app-chi.vercel.app). 감정 분석은 Claude haiku-4-5가 8개 감정 그룹과 자유형 구체 감정을 JSON으로 출력하고, 대화 기능 "대나무숲"은 pgvector 의미 검색, 유저 사실 메모리, 개인 지식 그래프를 도구로 쓰는 LangGraph 에이전트다. 감정 분류 프롬프트는 코드북과 394건 골든셋으로 두 차례 고쳤고, Accuracy 53.8%→71.6%, Macro F1 0.483→0.700, Kappa 0.469→0.670이 됐다. 위기 신호는 키워드 스크린과 LLM 판정을 OR로 결합하고, 위기 대화는 에이전트 루프에 들어가지 않게 했다. 초기 2주의 팀원 작업(바이오리듬 조사, 가상 유저 페르소나)을 제외한 본문 시스템은 모두 본인이 만들었다.

## 1. 문제 정의

입력은 일기 한 편, 출력은 네 가지다.

| 출력 | 형태 |
|---|---|
| 감정 분석 | `primary_group`(8개 중 1개), `specific_emotion`(자유형), `intensity`(1~10), `mixed`, `summary`, `risk_level`, `confidence` |
| 행동 가이드 | 행동 제안 2~3개, 강도 7 이상이면 휴식 메시지 |
| 대화 응답 | 대나무숲 페르소나의 3~5문장 반말 응답 |
| 위기 안내 | `none / caution / high`와 상담기관 안내 |

8개 그룹(긍정, 슬픔, 분노, 불안, 무기력, 외로움, 혼란, 복합)은 통계에, 자유형 감정("설레는 불안")은 화면에 쓴다.

평가 지표(`scripts/evaluate_emotion_analysis.py`):

```text
Accuracy      = 정답 그룹 일치율 (파싱 실패·8개 밖 예측은 '무효'로 오답 처리)
Macro F1      = 8개 그룹 F1의 단순 평균
Cohen's Kappa = (p_o - p_e) / (1 - p_e)
강도          = MAE, Pearson r
```

그룹 분포가 고르지 않아(긍정 85, 외로움 25) Macro F1과 그룹별 재현율을 채택 기준으로 했다.

제약은 1인 파트타임, 무료 인프라(Render 슬립, Voyage 3 RPM), 사용자 간 일기 격리, 원문 외부 유출 금지, 위기 신호 재현율 우선이다.

## 2. 데이터

평가에 실사용자 데이터는 쓰지 않았다.

| 데이터 | 크기 | 용도 |
|---|---|---|
| 시드 일기 | 가상 유저 100명 × 90일 = 9,000편 | dev DB, RAG 검색 시연 |
| 노이즈 일기 | 1,061편 (ultra_short 356 / slang_typo 282 / mixed_emotion 213 / factual_no_emotion 210) | 1차 평가셋, 골든셋 후보 |
| 골든셋 v1 | 394편 (slang_typo 129 / factual 124 / ultra_short 95 / mixed 46) | 프롬프트 채택 기준 |

노이즈 일기는 실제 입력처럼 지저분한 네 유형으로 만들었다. 정답 라벨은 생성 LLM이 붙인 것이라, 이 라벨로 프롬프트를 고치면 생성 LLM의 판단에 맞추게 된다. 그래서 골든셋을 따로 만들었다.

```text
노이즈 1,061 → 그룹 × 유형 층화 추출 400 (seed 42)
  ├─ 사람 검수 28
  ├─ 코드북 규칙을 넣은 심판(Claude) 재판정 → 생성 라벨과 일치 236 (llm_consensus)
  └─ 분쟁 133 → 세 번째 모델(GPT 계열) 판정 → 2:1 다수결 130 (llm_majority)
                                              3자 불일치 3 제외
  → 판정 실패 3 제외 → 골든셋 394
```

판정 기준은 `docs/emotion-codebook.md`에 고정했다. 핵심은 경계 규칙이다.

| 규칙 | 내용 |
|---|---|
| R1 | 복합은 최후의 선택지. 하나가 조금이라도 우세하면 그 그룹 |
| R2 | 결핍 대상이 사람이면 외로움, 잃어버린 것이면 슬픔, 에너지면 무기력 |
| R3 | 사실 나열 일기도 가장 가까운 그룹을 강도 1~3으로 |
| R4 | 두려운 대상이 명확하면 불안, 모르는 상태 자체면 혼란 |
| R5 | 표면과 이면이 팽팽하면 표면 감정 우선 |
| R6 | 상실 사건이 핵심이면 기본값은 슬픔 (3회차 추가) |

임베딩은 Voyage `voyage-3.5`(1024차원)로, 일기는 `document`, 질의는 `query`로 비대칭 임베딩한다. 청킹 없이 일기 한 편을 벡터 하나로 저장한다(앞 16,000자).

## 3. 방법

### 3.1 전체 파이프라인

```text
[POST /diaries]
  일기 저장
  ├─ 감정 분석 (Claude haiku-4-5, JSON 파싱 실패 시 1회 재시도)
  │    └─ 위기 감지: keyword_screen OR LLM risk_level, confidence < 0.6 로그
  ├─ 행동 가이드 (감정 JSON 입력)
  └─ BackgroundTasks: Voyage 임베딩 → 지식 그래프 추출 → "달라진 점" 한 줄

[POST /chat]
  최근 30개 메시지 + 최근 7일 일기(수면·컨디션·감정 점수)
  ├─ 위기 가드: keyword_screen != none → 도구 없는 1회 호출 + 상담기관 안내
  ├─ 1차 v2 LangGraph: 분류 → 전문가(전략 노트, 도구) → 보이스
  ├─ 2차 v1: 단일 tool use 루프 (최대 5회)
  └─ 3차 레거시: 코드가 pgvector 검색 → 프롬프트 주입 → 1회 호출
```

감정 분석은 저장 응답에 바로 보여야 해서 동기로, 임베딩과 그래프 추출은 백그라운드로 두었다. 대화는 상위 경로가 예외를 던지면 다음 경로가 받는다. LLM은 401/403·크레딧 소진일 때만 GPT-4o-mini로 전환한다.

### 3.2 감정 분석 프롬프트

분석과 공감 메시지를 한 번의 호출에서 JSON으로 받는다. 프롬프트의 `[분류 규칙]`은 코드북 경계 규칙을 옮긴 것이고, 나머지 섹션은 관찰된 실패 유형에 대응한다. "먹고 잤다"를 "못 먹고 잤구나"로 바꾸는 공감 문장에는 사실 왜곡 금지 규칙을, "테스트" 같은 입력에 대화체로 답해 JSON이 깨지는 문제에는 모든 입력을 일기로 취급하라는 규칙을 넣었다. 긴 음성 일기는 마지막까지 남은 감정을 primary로, 거쳐 간 감정을 `mixed`로 두게 했다.

파싱이 끝내 실패해도 키워드 위기 스크린은 실행한다. 행동 가이드는 감정 JSON을 입력으로 받는 두 번째 호출이다.

### 3.3 위기 감지

`services/risk_detection.py`는 high 9개, caution 9개 정규식으로 원문과 공백 제거본("죽 고 싶 다")을 모두 검사하고, LLM 판정과 높은 쪽을 택한다.

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

감지 시 상담기관(109, 1577-0199, 1388) 안내만 하고 자동 조치는 하지 않는다. 패턴, 비유 표현("짜증나 죽겠다"는 none), 결합 규칙은 CI 회귀 테스트로 검증한다.

### 3.4 RAG 장기기억

별도 벡터 DB 없이 `diaries.embedding vector(1024)`에 HNSW(`vector_cosine_ops`) 인덱스를 걸었다. 같은 테이블에서 `user_id`로 먼저 좁히므로 사용자 격리가 SQL 필터 하나로 강제된다.

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

대화용 검색은 top-3, 최근 7일 제외(이미 컨텍스트에 있음), 코사인 거리 0.65 초과 결과 폐기다. 0.65는 경험적 초기값이다.

벡터 검색은 특정 인물과의 관계 변화 같은 질문에 약하다. 그래서 일기 저장 시 LLM이 인물·사건·장소·주제·루틴과 관계를 추출해 `graph_nodes`, `graph_edges`에 쌓는다. 기존 노드 목록을 프롬프트에 주어 "여친"과 "여자친구"를 한 노드로 묶는다. 대화 중 알게 된 일정·관계는 `user_facts`에 저장해 최근 20개를 프롬프트에 넣는다.

### 3.5 대화 에이전트

v1은 Claude tool use 루프 하나다. 도구는 `search_diaries`, `get_emotion_history`, `get_sleep_pattern`, `explore_graph`, `save_user_fact`이고, 루프 최대 5회, 도구 결과 4,000자 제한이다.

v2는 LangGraph로 분류, 전문가, 보이스 노드를 연결했다. 전문가(침잠계·격동계·혼란계·일상계)는 사용자에게 말하지 않고 "진짜 감정 / 접근 / 피할 것 / 참고 기억" 전략 노트만 쓴다. 페르소나는 보이스 노드에만 있어서 분류가 틀려도 말투가 유지된다.

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

위기 판정은 그래프 진입 전에 키워드로 결정하고, 위기 경로의 상담기관 안내는 모델이 아니라 코드가 붙인다.

### 3.6 운영

| 항목 | 구현 |
|---|---|
| 레이트 리밋 | slowapi, 토큰 해시 기준 전체 120/분, LLM 엔드포인트 10/분 |
| 비용 추적 | 모든 Claude·GPT·Voyage 호출 토큰을 `api_usage`에 `purpose`별 기록 |
| 개인정보 | Sentry `send_default_pii=False`, production에서 LangSmith 환경변수 강제 제거(테스트 검증) |
| 스키마·CI | Alembic(baseline 포함 14개 리비전), GitHub Actions pytest 14개 + 프론트 빌드 |
| 스케줄 | Actions cron으로 일일 푸시, 시간별 리마인더, 주간 요약, keep-alive |


## 4. 실험

### 4.1 실험 설정

- 평가셋: 골든셋 394건, 버전당 1회 실행. 앱의 `analyze_emotion()`을 그대로 호출해 재시도·후처리를 포함했다.
- 모델: `claude-haiku-4-5-20251001`, `max_tokens=2048`, temperature 기본값.
- 버전: baseline(코드북 규칙 반영 전), v2(R1~R5와 confidence 반영), v3(R6 추가).
- 수치는 레포의 결과 파일 3개로 재계산했고 커밋 기록과 일치했다.

### 4.2 결과

| 지표 | baseline | v2 | v3 |
|---|---|---|---|
| Accuracy | 53.8% | 69.0% | **71.6%** |
| Macro F1 | 0.483 | 0.649 | **0.700** |
| Kappa | 0.469 | 0.637 | **0.670** |
| 무효 예측 | 19 | 8 | 7 |
| 강도 MAE | 0.85 | **0.78** | 0.84 |

| 그룹별 재현율 (정답 수) | baseline | v2 | v3 |
|---|---|---|---|
| 긍정 (85) | 0.71 | 0.81 | 0.81 |
| 무기력 (72) | 0.71 | 0.82 | 0.85 |
| 불안 (69) | 0.46 | 0.65 | 0.55 |
| 분노 (43) | 0.40 | 0.56 | 0.47 |
| 혼란 (40) | 0.42 | 0.70 | 0.60 |
| 복합 (30) | 0.93 | 0.80 | 0.90 |
| 슬픔 (30) | 0.10 | 0.17 | 0.80 |
| 외로움 (25) | 0.16 | 0.72 | 0.76 |

노이즈 유형별 정확도는 baseline → v3에서 ultra_short 62.1→77.9%, slang_typo 53.5→72.1%, factual_no_emotion 47.6→66.1%, mixed_emotion 54.3→71.7%다.

baseline은 394건 중 130건을 복합으로 예측했고(정답 복합 30건), 외로움 재현율 0.16, 사실 나열 정확도 47.6%였다. R1~R3이 이 지점을 겨냥했고 v2에서 복합 예측은 54건이 됐다. v2의 슬픔은 30건 중 5건만 맞았다. 낙방·탈락 같은 상실 뒤의 자책이 무기력으로, 원망이 분노로, 다음 걱정이 불안으로 분류됐다. R6(상실 사건이면 슬픔, 이미 벌어진 일과 아직 오지 않은 일을 시제로 구분)를 넣자 슬픔 재현율이 0.80이 됐다.

### 4.3 비교 실험

| 변형 | 결과 | 판단 |
|---|---|---|
| baseline을 노이즈셋(LLM 라벨 1,061건)으로 평가 | Acc 68.0%, Kappa 0.608 (골든셋 53.8%, 0.469) | LLM 라벨 기준 점수가 높게 나옴. 채택 기준을 골든셋으로 변경 (68.0%는 커밋 보고값) |
| + 코드북 R1~R5 (v2) | Macro F1 +0.166, 외로움 R 0.16→0.72 | 채택 |
| + 슬픔 규칙 R6 (v3) | 슬픔 R 0.17→0.80, Macro F1 +0.051 | 채택 |
| R6의 부작용 | 분노 R 0.56→0.47, 불안 0.65→0.55, 혼란 0.70→0.60, MAE 0.78→0.84 | 슬픈 사용자를 화난 사용자로 읽는 쪽이 더 나쁘다고 보고 수용. 다음 루프 후보 |
| 키워드 스크린 단독 (골든셋 위기 양성 6건) | 검출 0/6, 음성 388건 중 오탐 2 | 양성이 모두 위기 단어 없는 사실 나열 일기라 키워드만으로는 부족. LLM 판정은 미측정 |
| 사람 전수 검수 | 28건에서 중단 | 하이브리드 검수로 전환 |

## 5. 한계와 향후 과제

1. 골든셋의 사람 검수 비중이 7%(28/394)다. 71.6%는 "코드북을 따르는 LLM 합의와의 일치율"에 가깝다. 실사용 일기 샘플링 검수는 아직 하지 않았다.
2. 같은 골든셋으로 프롬프트를 고치고 평가했다. 홀드아웃이 없어 v3에는 과적합이 섞였을 수 있고, 버전당 1회 실행이라 분산도 모른다.
3. RAG 검색과 에이전트 응답 품질을 정량 평가하지 않았다. 거리 임계값 0.65를 정하려면 질의별 정답 일기를 둔 Recall@k 평가셋이 필요하다.
4. 위기 감지 평가셋이 부족하다. 양성 6건뿐이고 LLM 판정은 기록되지 않는다.
5. Alembic baseline이 빈 리비전이라 새 DB는 별도 스크립트로 만들어야 하고, RLS는 미적용이다.

## 참고

- 개발 기록(티스토리): https://mynews76154.tistory.com/7
- 서비스: https://diary-ai-app-chi.vercel.app
- 모델·도구: Claude haiku-4-5, GPT-4o-mini, voyage-3.5, pgvector, LangGraph, FastAPI, React PWA, Supabase
