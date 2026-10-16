---
title: "공정위 의결서 RAG: 하이브리드 검색·이중 리랭커 RRF와 EXAONE 3.5 LoRA 생성"
description: "의결서 500건(31,877청크)에서 근거 청크 5개와 답변을 돌려주는 RAG를 8B 이하 로컬 LLM·재청킹 금지·30초 제한 아래 구성했다. 자체 평가셋 278문항 기준 R@5 0.6930, MRR 0.8007, 종합 67.76점."
date: 2026-10-16
lang: ko
key: kftc-rag
tags: [RAG, Hybrid Retrieval, Reranker, RRF, Contextual Chunking, EXAONE, LoRA]
tistory: https://mynews76154.tistory.com/1
---

## 초록

제2회 공정거래 AI·데이터 활용 공모전 Track 2(2026.04~05, 2인 팀)에서 공정거래위원회 의결서 500건을 대상으로 질의마다 근거 `chunk_id` 5개와 답변을 반환하는 RAG 시스템을 만들었다. 재청킹 금지, 8B 이하 로컬 LLM, 문항당 30초 규칙 때문에 검색 품질은 청크 경계가 아니라 색인 텍스트(맥락 prefix, 정규화, 표 문장화)로 올렸다. 검색은 BM25와 dense 임베딩 2종의 하이브리드 결과를 RRF로 합치고, 사전학습 리랭커 2종의 top-5를 다시 RRF로 합산한다. 생성은 EXAONE 3.5 7.8B(4bit nf4)에 attention·MLP 7개 모듈 LoRA를 붙이고, 1,500건으로 학습한 어댑터에서 5,000건을 이어 학습했다. 자체 평가셋 278문항 기준 R@5 0.6930, MRR 0.8007, BERTScore F1 0.7830, Token F1 0.4000, 종합 67.76점이다.

## 1. 문제 정의

입력은 자연어 질의, 출력은 근거 `chunk_id` 5개와 답변이다. 평가 서버는 Docker 컨테이너의 `POST /predict`로 질의를 보낸다. 검색 50점, 생성 50점이다.

```text
R@5   = |top5 ∩ gold| / |gold|        (문항 평균)
MRR   = 1 / (top5 안 첫 정답 순위), 없으면 0
Final = 0.35*R@5 + 0.15*MRR + 0.30*BERTScore_F1 + 0.20*TokenF1   (x100)
```

자체 측정의 BERTScore는 `klue/roberta-base`로 계산했다. 제약은 다음과 같다.

| 규칙 | 위반 시 |
|---|---|
| `chunk_id` 정확히 5개, 중복 불가 | 검색 0점 |
| 원본 `chunk_id`만 사용 (재청킹 금지) | 해당 청크 오답 |
| 문항당 30초 이내 | 문항 전체 0점 |
| 생성 LLM 8B 이하, 외부 API 금지, 오프라인 | 실격 |

평가 환경은 A100 80G 2장이며 임베딩 모델 크기에는 제한이 없다.

## 2. 데이터

| 항목 | 규모 | 비고 |
|---|---|---|
| 의결서 코퍼스 | 500건, 31,877 청크 | `DOC-{UUID}-CH-{번호}` |
| 표 청크 | 10,609 (33.3%) | `raw_text`에 `\|` 5개 이상 |
| 자체 평가셋 `qa_dataset.json` | 278문항 | 정답 청크 평균 2.04개(1~3), 정답 평균 110자 |
| 학습 QA v1 `train_qa.json` | 1,500건 | GPT-4o-mini 생성, 답변 평균 88자 |
| 학습 QA v6 `train_qa_v6.json` | 5,000건 | GPT-4o-mini, 페르소나 3종, 정답 청크 단일 30% / 인접 40% / 흩어짐 30% |
| 본평가 | 200문항 | 비공개 |

원문 `raw_text`는 보존하고 색인용 `text`만 다시 만들었다.

1. **Contextual prefix:** 청크 앞에 `[위반유형] [문서명] [섹션] [헤더]`를 붙인다.
2. **정규식 정규화:** 날짜 `2017.4.27.` → `2017년 4월 27일`, 금액 뒤 `(약 100만원)` 병기, 익명화 기호 `●○☆△`, `ㅇㅇㅇ` → `[비공개]`.
3. **표 문장화:** 표 청크를 GPT-4o-mini로 문장화하고 검증을 통과한 1,524건(14.4%)만 적용했다. 색인 시점에 한 번만 실행하므로 추론 시점의 외부 API 호출은 없다.

prefix의 근거는 실패 분석(`analysis/analyze_failures.py`)이다. 100자 미만 정답 청크의 실패율이 90%, 오답 중 정답의 인접 청크(CH±1, 2) 비율이 46.4%, 표 형태 청크 비율이 31.8%였다.

## 3. 방법

### 3.1 전체 파이프라인

```text
질의
 ├─ BGE-m3-ko FAISS      + BM25   (min-max 선형 결합)
 └─ Arctic-embed-ko FAISS + BM25   (min-max 선형 결합)
        ▼
 두 하이브리드 순위 RRF (k=60) → 후보 5 x pool
        ├─ mxbai-rerank-large-v2  후보 30 → top 5
        └─ bge-reranker-v2-m3     후보 40 → top 5
        ▼
 두 top-5 RRF (k=60) → chunk_id 5개
        ▼
 EXAONE 3.5 7.8B (4bit nf4) + LoRA v6_c, greedy, 128 tokens → 답변
```

- 하이브리드: 임베딩 리트리버마다 BM25와 dense 점수를 min-max 정규화해 선형 결합한다.
- 앙상블·리랭커 합산: `score = Σ weight / (k + rank + 1)`. 점수 스케일이 다른 두 리랭커를 순위만으로 합치기 위해 RRF를 썼다(`scripts/rrf_merge_rerankers.py`).
- 생성: 상위 5개 청크를 `[1]`~`[5]`로 번호 매겨 컨텍스트로 넣는다.

| 구성 요소 | 담당 |
|---|---|
| 임베딩 모델 비교·선정, reranker 파인튜닝(v2~v6) | 팀원 |
| 실패 분석, Contextual Chunking, 인덱스 정규화, 표 문장화, 리랭커 결과 RRF 합산 | 본인 |
| LoRA 학습 데이터, EXAONE LoRA 학습, 생성 평가 | 본인 |

### 3.2 Contextual Chunking

Anthropic Contextual Retrieval의 아이디어를 LLM 호출 없이 메타데이터 결합으로 줄였다. 같은 함수를 FAISS 빌드(`shared/data/loader.py`)와 BM25 빌드(`shared/indexing/build_bm25.py`)가 공유해 두 인덱스가 같은 텍스트를 본다. 위반유형은 `A; A; B`처럼 반복 집계된 문자열이라 중복을 제거한다.

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

### 3.3 표 문장화 검증

변환 중 금액이나 날짜가 바뀌면 근거가 틀린다. 원문 숫자 토큰 80% 이상, 2글자 이상 한글 단어 60% 이상이 보존된 경우만 채택하고 실패나 API 오류는 원문으로 fallback한다.

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

EXAONE은 MLP 모듈 이름이 `c_fc_0`, `c_fc_1`, `c_proj`(LLaMA의 gate/up/down_proj)라서, 학습 전에 `named_modules()`로 target 존재를 확인하고 누락 시 경고한다. trainable 파라미터는 attention 4모듈 13,631,488(0.174%), MLP 포함 7모듈 41,943,040(0.534%)이다. 학습 데이터는 EXAONE 채팅 포맷(시스템 지시, 참고 문서 + 질문, 답변)이고 시스템 지시는 "1~2문장 100자 내외, 마크다운·메타 코멘트 금지"로 고정했다. 최종 어댑터 v6_c는 7모듈 v1 어댑터를 학습 가능 상태로 불러와 v6 5,000건을 이어 학습했다.

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

이어 학습 경로는 `LoraConfig`를 새로 만들지 않으므로 r, alpha, dropout은 시작 어댑터 값(16, 32, 0.05)을 따른다.

## 4. 실험

### 4.1 실험 설정

- 평가셋: 자체 평가셋 278문항, 단일 holdout. 개발·측정 GPU는 RTX 5070 Ti(16GB) 1장.
- 검색: `top_k 5`, chunk 기준 평가, 리랭커 후보 수 `5 x pool`.
- LoRA 공통: r 16, alpha 32, dropout 0.05, batch 1 x grad accum 8, warmup 30, max_seq 1536, `paged_adamw_8bit`, fp16.
- v1/7모듈: 2 epoch, lr 2e-4, 1,500건(40분 / 48.4분). v6_c: 1 epoch, lr 1e-4, 5,000건, 약 122분.
- 추론: greedy, max_new_tokens 128, few-shot 0, 기본 후처리(마크다운·번호·메타 문구 제거).

#### 평가 프로토콜

- 학습 QA(v1, v6)는 별도 생성했고 평가셋과 질문 중복은 0건이다.
- v6 생성기는 평가셋 정답 청크 566개를 시드 후보에서 제외한다(`scripts/generate_train_qa_v6.py`).
- few-shot 예시는 `train_qa.json` 풀에서만 뽑는다(`--few_shot_qa_path`).
- 정답 위반유형을 주입하는 상한 측정은 `--use_oracle_vt_filter`와 `*_oracle.json`으로 분리하고 채택 판단에 쓰지 않는다.

이 원칙은 4월 30일 이후 적용했다. 그 전에 임베딩 contrastive FT(본인, 평가셋 278문항을 triplet 질문으로 사용)와 reranker FT(팀원, 평가셋 222문항으로 학습)에 평가셋 질문이 들어간 것을 확인해, 그 상태의 측정값 72.96점을 무효 처리하고 reranker를 사전학습 모델로 되돌렸다.

### 4.2 결과

| 단계 | 지표 | 값 | 환산 |
|---|---|---|---|
| 검색 | R@5 / MRR | 0.6930 / 0.8007 | 36.27 / 50 |
| 생성 | BERTScore F1 / Token F1 | 0.7830 / 0.4000 | 31.49 / 50 |
| 종합 | | | 67.76 / 100 |

top-5에 정답이 1개 이상 든 문항 비율은 0.9317, 생성 시간은 문항당 평균 5.66초다. 실험 로그의 점수 변화는 다음과 같다(무효 측정 제외).

| 시점 | 변경 | 검색 | 생성 | 종합 |
|---|---|---|---|---|
| 4/24 | 기준선 (LoRA 없음) | 31.38 | 28.53 | 59.91 |
| 4/27 | + LoRA v1 | 32.92 | 30.90 | 63.82 |
| 4/29 | + 7모듈 LoRA | 32.92 | 31.12 | 64.04 |
| 4/29 | + 임베딩 측 개선 (팀원) | 36.02 | 31.12 | 67.14 |
| 5/3 | 사전학습 reranker로 복귀 | 35.95 | 31.12 | 67.07 |
| 5/4 | 정규화 + 이중 리랭커 RRF | 36.27 | 31.12 | 67.39 |
| 5/5 | + LoRA v6_c | 36.27 | 31.49 | 67.76 |

4/27~4/29의 검색 32.92는 평가셋 질문으로 학습한 임베딩 FT가 적용된 상태의 값이다.

### 4.3 비교 실험

**검색 (1) 초기 설정(`top_k 10`, bge 리랭커, pool 8).**

| 구성 | R@5 | MRR | 시간(s) |
|---|---|---|---|
| 5모델 앙상블, Oracle | 0.5402 | 0.6919 | 29.38 |
| 2모델(bgem3ko + arctic), Oracle | 0.5420 | 0.6955 | 2.98 |
| 2모델 + Contextual, Oracle | 0.5947 | 0.7349 | 4.65 |
| 2모델 + Contextual (FAISS + BM25) | 0.5815 | 0.7350 | 3.33 |
| 2모델 + Contextual, pool 20 | 0.5647 | 0.7307 | 7.28 |
| 2모델 + Contextual + 위반유형 분류기 필터 | 0.4844 | 0.6148 | 11.82 |
| 2모델 + Contextual + contrastive 임베딩 FT (본인, 무효) | 0.5815 | 0.7291 | |

5모델과 2모델은 R@5가 같고 시간은 약 10분의 1이다. Contextual Chunking은 같은 2모델·Oracle 조건에서 +5.27%p였다(커밋에 적힌 +8.45%p는 5모델 무필터 0.4934 대비라 모델 수 변화가 섞여 있다). 분류기(정확도 62.95%)는 오분류가 정답을 후보에서 지워 9.71%p 하락했다.

**검색 (2) mxbai 후보 수(`top_k 5`).**

| pool | 4 | 6 | 8 | 10 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| R@5 | 0.6871 | 0.6841~0.6960 | 0.6823 | 0.6793 | 0.6703 | 0.6727 | 0.6703 |
| MRR | 0.8001 | 0.7944~0.8052 | 0.7964 | 0.8005 | 0.7931 | 0.7919 | 0.7844 |
| 시간(s) | 0.92 | 1.46~1.71 | 1.72 | 2.22 | 2.82 | 3.45 | 4.01 |

pool 6은 같은 설정 4회 측정값의 범위다. pool 4와 8은 이 범위와 비슷하고, 10 이상에서 R@5가 내려간다.

**검색 (3) 리랭커 구성과 정규화.**

| 구성 | R@5 | MRR |
|---|---|---|
| mxbai pool 6, 정규화 전 (최고 회차) | 0.6960 | 0.7944 |
| mxbai pool 6, 정규화 후 | 0.6924 | 0.7978 |
| bge pool 8, 정규화 후 | 0.6745 | 0.7809 |
| mxbai + bge RRF, 정규화 후 (최종) | 0.6930 | 0.8007 |
| reranker FT v6 (팀원), pool 8 | 0.6487 | 0.7708 |
| 3모델(+ Qwen3-Embedding INT8, 팀원), mxbai pool 6 | 0.6805 | 0.7936 |

정규화(R@5 -0.36%p, MRR +0.34%p)와 이중 RRF(mxbai 단독 대비 MRR +0.29%p)는 pool 6 측정 변동 폭 안의 변화다.

**생성 (1) 학습 데이터와 방식.** 실험 로그 값, 당시 검색 결과 입력.

| 변형 | 내용 | BS F1 | Tok F1 | 점수 |
|---|---|---|---|---|
| v1 | 자유형식 1,500건, attention 4모듈 | 0.7776 | 0.3788 | 30.90 |
| v2 | 정형 단문 강제(평균 54자), epoch 5, r 32 | 0.6618 | 0.1941 | 23.73 |
| v3 | 정형 시작 + 자유 설명(평균 110자) | 0.6752 | 0.2085 | 24.43 |
| CPT | v1 위에 청크 원문 4,231건 | 0.7752 | 0.3719 | 30.69 |
| Mixed | 원문 74% + QA 26%, base에서 | 0.7605 | 0.3546 | 29.91 |
| 7모듈 | v1 데이터, MLP 추가 | 0.7807 | 0.3852 | 31.12 |
| Hard-example | 약점 위반유형 QA 400건 추가 | 0.7796 | 0.3811 | 31.01 |
| v6_a / v6_b / v6_f | v6 5,000건 base에서 신규 학습 | 0.7761 / 0.7752 / 0.7792 | 0.3820 / 0.3819 / 0.3873 | 30.92 / 30.90 / 31.12 |

v1 기준 few-shot 3은 31.05(+0.15)였고, 컨텍스트 8청크(30.90), 2문장 절단(30.79), max_new_tokens 150(31.05)은 개선이 없었다.

**생성 (2) 최종 검색 결과 고정.**

| 구성 | BS F1 | Tok F1 | 점수 | s/문항 |
|---|---|---|---|---|
| 7모듈 v1 어댑터 | 0.7827 | 0.3979 | 31.44 | 4.75 |
| v6_c (v1에서 이어 학습) | 0.7830 | 0.4000 | 31.49 | 5.66 |
| v6_c + few-shot 3 + 프롬프트 v2 | 0.7734 | 0.3885 | 30.97 | 6.24 |
| v6_c + few-shot 3 + 후처리 strong | 0.7801 | 0.3936 | 31.28 | 6.21 |
| v6_c + few-shot 3 + temperature 0.3 | 0.7796 | 0.3910 | 31.21 | 6.23 |
| v6_c + 위 옵션 전부 | 0.7229 | 0.2968 | 27.62 | 6.63 |

로그는 v6_c 효과를 +0.37점(31.12 → 31.49)으로 적었지만 31.12는 이전 검색 결과의 값이다. 같은 검색 입력에서는 +0.05점이다. dropout 0.1을 지정한 v6_g가 v6_c와 지표까지 같았던 것은 이어 학습 경로가 시작 어댑터 설정을 쓰기 때문이다.

## 5. 한계와 향후 과제

- **단일 평가셋:** 278문항 하나로 모든 채택을 결정했다. pool 6의 회차 간 R@5 변동이 1.19%p인데 정규화, 이중 RRF, v6_c의 채택 근거는 이보다 작다. 개발용과 최종 확인용을 나누고 반복 측정으로 분산을 확인해야 한다.
- **청크 단위 분리 불완전:** 평가셋 정답 청크 566개 중 70개가 v1 학습 QA의 시드 청크이고, 76개가 v6 학습 QA의 정답 청크 목록에 있다. v6는 시드만 걸러서 인접·흩어짐 모드로 붙는 청크는 남는다.
- **BM25 가중치 미전달:** 실행 인자는 `--bm25_weight 0.3`이지만 bgem3ko·arctic의 HybridRetriever는 앙상블 로드 시 이 값을 받지 않아 `DEFAULT_ALPHA = 0.2`로 결합한다. 보고 수치는 이 상태의 값이다.
- **종단 지연 미검증:** FastAPI 라우터는 스텁이고 지연은 단계별로만 쟀다(리랭커 1.71초, 1.88초, 생성 5.66초). 서빙 경로에서 30초 제한을 종단으로 확인해야 한다.
- **미시도:** v6_c 가중치는 레포에 없어 재학습이 필요하다. DPO, 답변 후보 다중 생성 후 선택, 문장 단위 컨텍스트 압축은 시도하지 못했다.

## 참고

- 코드: https://github.com/tjehdgus/KFTC
- 개발 기록(티스토리): https://mynews76154.tistory.com/1
- 모델: EXAONE 3.5 7.8B Instruct, dragonkue/BGE-m3-ko, dragonkue/snowflake-arctic-embed-l-v2.0-ko, mxbai-rerank-large-v2, bge-reranker-v2-m3, klue/roberta-base
- 방법: BM25, Reciprocal Rank Fusion, LoRA, QLoRA (NF4), Anthropic Contextual Retrieval
