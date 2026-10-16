---
title: "라이프로그 센서 기반 수면 7지표 예측: 9소스 타겟별 블렌드와 forward-chaining 재가중"
description: "폰·워치 센서 하루치로 그날 밤 수면 7개 이진 지표를 예측했다. 9개 이질 소스를 타겟별로 블렌드하고 S1/S4 가중치를 시간순 폴드로 다시 골라 Private Log-Loss 0.57849(22위)를 기록했다."
date: 2026-10-16
lang: ko
key: etri-sleep
tags: [Lifelog, Sleep Prediction, Ensemble, Time-series CV, Distribution Shift, SigLIP]
tistory: https://mynews76154.tistory.com/15
---

## 초록

제5회 ETRI 휴먼이해 인공지능 논문경진대회(2026.06) 과제는 스마트폰·워치 센서 12종의 하루 기록으로 그날 밤 수면의 7개 이진 지표를 예측하는 것이다. 데이터는 피험자 10명, 학습 450박, 평가 250박이고 평가 밤이 학습 밤보다 뒤쪽 날짜에 몰려 있다. 트리, 시퀀스 DL 4종, 동결 SigLIP 임베딩, 수면 구간 재구성, 날짜 이웃 라벨 등 9개 소스를 타겟별 greedy 블렌드로 결합했고, S1/S4 가중치는 날짜순 폴드로 다시 골랐다. Q1/Q2에는 비지도 일상상태(NMF) 라우팅과 피험자내 pairwise 순위 신호를 logit 잔차로 더했다. 블렌드 OOF 0.58431, 최종 Public 0.5584, Private 0.57849(22위)이며 논문은 ICTC 2026에 채택됐다(게재는 하지 않음). 팀 이름으로 제출했고 레포의 코드와 논문은 본인이 작성했다.

## 1. 문제 정의

- **입력**: `(subject_id, lifelog_date)` 단위 센서 원시 기록. 워치 심박·보행·조도, 폰 충전·활동·주변소리·BLE·GPS·조도·화면·앱사용·Wi-Fi.
- **출력**: 같은 행 `sleep_date` 밤의 7개 라벨 확률.

| 라벨 | 의미 | 기준 |
|---|---|---|
| Q1 / Q2 / Q3 | 수면질 / 피로 / 스트레스 (설문) | 개인 전체기간 평균 대비 |
| S1 / S2 / S3 / S4 | TST / 수면효율 / 입면지연 / WASO | NSF 가이드라인 충족 여부 |

- **지표**: 라벨별 log-loss 평균. 전부 0.5를 내면 0.693.

```text
L = (1/7) * sum_j [ -(1/N) * sum_i ( y_ij*log(p_ij) + (1-y_ij)*log(1-p_ij) ) ]
```

- **제약**: 같은 피험자가 학습·평가에 모두 등장하고 날짜만 분리된다. S1~S4는 입력에 없는 Withings 수면측정기에서 계산된 라벨이다. 하루 제출은 3회. 모든 확률은 `[0.02, 0.98]`로 클리핑했다.

## 2. 데이터

| 처리 | 내용 |
|---|---|
| 심박 | 타임스탬프마다 초단위 배열. 30~220bpm 밖 값 제거 후 mean/min/max/sd |
| 리스트형 센서 | 주변소리 8개 클래스 확률, GPS 평균 속도·위경도 범위, Wi-Fi·BLE 기기 수, 앱 사용시간 합 |
| 시간 윈도우 | 관측일 기준 all/day(9~18)/eve(18~24)/prebed(21~24), 밤 기준 night(0~7)/morning(6~10)/deep(0~4). 0~10시 기록은 전날 밤으로 귀속 |
| 통계 | mean/std/min/max/median, 누적형 변수는 sum 추가 |
| 피험자내 파생 | 컬럼마다 lag-1 차분, 3일 이동평균, EMA(span 3), 누적평균 |
| 달력 | 요일, 월, 주말, 피험자 번호 |

기본 집계 1,015개에 파생 4종과 달력 4개가 붙어 피처는 5,079개다(노트북 출력 `FC5079`). 파생 피처는 Q 라벨의 "개인 평균 대비" 정의에 맞춰 "그 사람 평소 대비 오늘"을 표현한다.

## 3. 방법

### 3.1 전체 파이프라인

```text
센서 parquet 12종 → 집계 + 피험자내 파생 (5,079 피처)
  → 베이스 소스 9종 (OOF / test)
      ├─ tree        LGBM + CatBoost + ExtraTrees, 중요도 top-150, pseudo-label
      ├─ prior       피험자별 라벨 평균
      ├─ GRU / BiLSTM / TCN / Attn   CNN + 시퀀스 인코더 + 피험자 임베딩, 8 seed 평균
      ├─ SigLIP      센서 이미지 → 동결 SigLIP-Large → LGBM
      ├─ SleepTree2  수면 구간 재구성 피처 + tree
      └─ Neighbor    날짜 인접 라벨 지수가중 평균
  → 타겟별 greedy 블렌드 (Q1/Q2는 daystate 소스 추가)
  → S1/S4 가중치 forward 폴드로 재선택
  → Q1/Q2 logit 잔차: daystate(α=0.45) → pairwise(α=0.15)
```

소스는 단독 성능보다 오류 패턴의 차이를 기준으로 골랐다.

### 3.2 베이스 소스

| 소스 | 구성 |
|---|---|
| tree | 폴드 안에서 ExtraTrees 중요도 상위 150개 선택. `p = 0.4*CatBoost + 0.3*LGBM + 0.3*ET`. LGBM은 num_leaves 7, min_child_samples 40, reg_lambda 5, CatBoost는 depth 4, l2 8. 1차 예측에서 `max(p,1-p) >= 0.85`인 test 행을 하드 라벨로 추가해 재학습 |
| prior | 폴드 학습 구간의 피험자별 라벨 평균. 블렌드 탐색의 시작점 |
| DL 4종 | 6채널(hr, step, screen, charge, light, activity) × 24시간 입력. GRU, BiLSTM, dilation 1/2/4 TCN, 4-head attention. 피험자 임베딩 8차원, 7헤드 BCE, AdamW lr 2e-3, dropout 0.4, 250에폭. seed 0~7 예측 평균(DLens) |
| SigLIP | 12채널 × 24시간을 256×256으로 보간해 `google/siglip-large-patch16-256`에 입력. 가중치 동결, pooler 출력으로 LGBM 학습 |
| SleepTree2 | 아래 재구성 피처 15개를 5,079 피처에 붙여 tree와 같은 방식으로 학습 |
| Neighbor | `p_ij = sum_m w_im*y_mj / sum_m w_im`, `w_im = exp(-|날짜차| / 3일)`. 같은 피험자 train 행만, OOF는 자기 자신 제외 |

**수면 구간 재구성.** S1~S4가 임계 처리하는 양을 직접 근사하려는 소스다. 18시~익일 12시 1,080분 그리드에서 다음 규칙을 적용했다.

```text
worn   = 보간 심박이 있는 분
rest   = (step <= 1) & worn, 11분 이동평균 > 0.5 → 최장 연속 블록(30분 이상)이 주수면 후보
asleep = 블록 내부 & HR < median(HR_worn) + 5, 7분 이동평균 > 0.4
TST = asleep 분 수, SE = TST/TIB, SOL = onset - 잠자리 시작(블록 시작과 첫 충전 중 이른 쪽)
WASO = (wake - onset + 1) - asleep 분 수
플래그: TST >= 420, SE >= 0.85, SOL <= 30, WASO <= 20
```

임계값은 코드에서 정한 근사다. 재구성 v1(TST 중앙값 256분), v3(219분)는 너무 엄격해 v2를 썼다.

**Neighbor의 근거.** 라벨에 날짜 자기상관이 있었고(Q1 +0.27, Q2 +0.30, S +0.15~0.24), test 밤이 train 밤과 시간상 섞여 있어 가까운 날 라벨이 센서와 직교하는 단서가 된다.

### 3.3 폴드 설계

베이스 소스 OOF는 피험자별로 날짜순 행을 10개 블록으로 나누고 폴드 f가 블록 f와 f+5를 검증으로 쓰는 5폴드(`folds_sh`)로 만들었다. 모든 피험자의 앞·뒤 구간이 한 번씩 가려져 무작위 셔플보다 인접일 누수가 적다. 블렌드에는 무작위 5폴드와 아래 forward 폴드를 쓴다.

`solution_full.py`
```python
rs=allidx[np.argsort(sd)]                                   # 날짜순 정렬
# ★forward-chaining CV: 앞 날짜로 학습→뒤 날짜로 검증(test의 시간이동을 모사). 시작점 0.5~0.8을 미끄러뜨리며 4폴드
forward_folds=[(rs[:int(ntr*f)],rs[int(ntr*f):int(ntr*min(f+0.2,1.0))]) for f in [0.5,0.6,0.7,0.8]]
```

### 3.4 타겟별 greedy 블렌드

타겟마다 음수 없는 합 1 가중치를 찾는다. prior=1에서 시작해 각 소스 가중치를 ±0.1, ±0.05 바꾼 후보 중 log-loss를 가장 낮추는 하나를 채택하기를 최대 40회 반복하고, 다음 폴드는 앞 폴드 결과에서 이어간다.

`solution_full.py`
```python
    for tr,va in foldgen:
        cur=log_loss(y[tr,j],C(sum(greedy_weights[i]*oof_list[i][tr,j] for i in range(num_sources))),labels=[0,1])
        for _ in range(40):                                 # coordinate-descent: 소스 가중치를 ±0.05/0.1씩 바꿔가며 logloss 최소화
            bb=(cur,None)
            for i in range(num_sources):
                for dd in (.1,-.1,.05,-.05):
                    w2=greedy_weights.copy()
                    w2[i]=max(0,w2[i]+dd)                   # 음수 금지 후 합=1로 정규화
                    ss=w2.sum()
                    if ss==0: continue
                    w2/=ss
                    ll=log_loss(y[tr,j],C(sum(w2[k]*oof_list[k][tr,j] for k in range(num_sources))),labels=[0,1])
                    if ll<bb[0]: bb=(ll,w2)
            if bb[1] is None: break
            greedy_weights,cur=bb[1],bb[0]
```

거친 스텝과 prior 앵커를 쓴 것은 연속 최적화가 LB에서 크게 악화됐기 때문이다(4.3). 7개 타겟을 무작위 폴드 가중치로 블렌드한 뒤 S1·S4만 forward 폴드로 같은 탐색을 다시 해 가중치를 교체한다.

### 3.5 Q1/Q2 보정

- **daystate**: train+test 700일을 30분 bin × 6채널(주변소리 엔트로피, GPS 이동량, 활동, 화면, Wi-Fi·BLE 수)로 만들어 NMF 8성분으로 분해했다. 상태 점유율, 전이 수, 첫/마지막 활동 시점, 저녁 연속성, 루틴 이탈도, 엔트로피 14개와 피험자내 z-score 14개를 LGBM 소스로 썼다. 라벨을 쓰지 않는 transductive 표현이다.
- **pairwise**: 피험자별 (양성 밤, 음성 밤) 쌍을 최대 400개 뽑아 피처 차로 L2 로지스틱 회귀(C=0.05)를 학습했다. 점수를 그 피험자 train 점수 안의 순위 `rank`로 바꾸고 `sigmoid(0.7*logit(prior) + 0.3*logit(rank))`를 만든다.

`solution_full.py`
```python
    base = C(_sig(_logit(blend[:, j]) + 0.45*(_logit(ds_p[:, j]) - _logit(blend[:, j]))))
    blend2[:, j] = C(_sig(_logit(base) + 0.15*(_logit(_pair_src[j]) - _logit(base))))
```

## 4. 실험

### 4.1 실험 설정

- 소스 OOF는 `folds_sh` 5폴드(Neighbor만 leave-one-out). 블렌드 OOF는 무작위 5폴드에서 학습 구간으로 가중치를 정하고 검증 구간에 적용했다.
- 개발 중 블렌드 비교는 가중치 탐색을 바깥 폴드와 분리한 nested 점수로 했다.
- RTX 5070 Ti 16GB, torch 2.11. 전체 실행 597초(노트북 출력).

### 4.2 결과

| 소스 | OOF | 소스 | OOF |
|---|---|---|---|
| prior | 0.6270 | TCN | 0.6153 |
| tree | 0.6145 | Attn | 0.6193 |
| SleepTree2 | 0.6150 | daystate | 0.6290 |
| GRU | 0.6238 | SigLIP | 0.6951 |
| BiLSTM | 0.6252 | **9소스 블렌드** | **0.58431** |

라벨별 OOF(`assets/result_srcll.png`):

| 소스 | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|---|---|
| tree | 0.61 | 0.69 | 0.70 | 0.54 | 0.57 | 0.53 | 0.66 |
| prior | 0.66 | 0.69 | 0.69 | 0.58 | 0.58 | 0.54 | 0.65 |
| GRU | 0.66 | 0.68 | 0.68 | 0.59 | 0.58 | 0.55 | 0.62 |
| SleepTree2 | 0.60 | 0.69 | 0.70 | 0.53 | 0.57 | 0.55 | 0.66 |
| Neighbor | 0.67 | 0.62 | 0.67 | 0.65 | 0.62 | 0.60 | 0.70 |

라벨마다 최강 소스가 다르고(S1·Q1 SleepTree2, S4 GRU, Q2 Neighbor), Q2·Q3는 chance 근처다. 최종 성적은 Public 0.5584(약 26위), Private 0.57849(22위)이고 Private 1위는 0.56758이다.

### 4.3 비교 실험

forward 재가중 전후 가중치(`assets/result_forward.png` 근사값):

| 타겟 | 폴드 | tree | SigLIP | SleepTree2 | GRU | TCN | Neighbor |
|---|---|---|---|---|---|---|---|
| S1 | random | 0.33 | 0.11 | 0.56 | 0 | 0 | 0 |
| S1 | forward | 0.42 | 0.18 | 0.41 | 0 | 0 | 0 |
| S4 | random | 0.05 | 0 | 0 | 0.90 | 0 | 0.04 |
| S4 | forward | 0 | 0.05 | 0 | 0.67 | 0.10 | 0.19 |

구성 변형별 로컬 점수와 Public LB(`LEARNINGS.md`, `개발과정_팀원용.md`):

| 변형 | 로컬 | LB |
|---|---|---|
| 단순 격자 블렌드 | OOF 0.6119 | 0.6081 |
| scipy 9소스 연속 최적화 | OOF 0.5992 | 0.6231 (악화) |
| 고정 가중치 블렌드 | OOF 0.6112 | 0.61055 |
| 8소스(+이미지 CNN) greedy | OOF 0.5950 | 0.5994 |
| 10소스(+ResNeXt101, SigLIP) | OOF 0.5936 | 0.6016 (악화) |
| CNN 2개 → SigLIP + SleepTree2 | nested 0.6019 | 0.5960 |
| + Neighbor | nested 0.6003 | |
| + daystate 라우팅 + pairwise | | 0.5596 |
| + S1/S4 forward 재가중(강도 단계 증가) | | 0.5588 → 0.5584 |
| 타겟별 후진제거 + TabPFN | 같은 폴드 선택 0.5917, 누수 제거 0.6590 | 기여 없음 |
| 피험자별 TabPFN | nested 0.81 | 기여 없음 |
| NSF 임계 1D 보정(S1) | 0.639 (SleepTree2 0.530) | 블렌드 악화 |
| 야간 HRV 추가 | 블렌드 0.6019 → 0.6024 | |
| 수면 재구성 v3 | S 평균 0.581 (v2 0.573) | |

LB로 전이된 변화는 소스 다양성, Q 정의에 맞춘 보정, DL seed 평균, 시간순 재가중이었다. OOF는 좋아지지만 기존 예측에서 크게 벗어나는 후보는 대부분 LB에서 악화됐다.

## 5. 한계와 향후 과제

1. **S1~S4 상한.** 라벨 원천인 Withings 데이터가 없어, 재구성 규칙을 바꿔도 S 라벨 log-loss는 0.53~0.58에 머물렀다.
2. **forward 재가중의 평가.** 탐색 함수는 폴드의 `tr` 구간으로만 손실을 계산하고 `va`는 쓰지 않아, 실제로는 날짜순 prefix(50%→80%) 순차 재적합이다. 효과는 Public LB 반복 제출로만 확인했으므로 검증 블록 손실을 쓰는 nested forward 평가가 필요하다.
3. **전역 날짜 정렬.** forward 폴드가 피험자 구분 없이 정렬돼, 기록 기간이 다른 피험자가 학습·검증 구간에 치우칠 수 있다. 피험자별 시간 분할이 test 구조에 더 가깝다.
4. **OOF 낙관 편향.** DL은 검증 폴드 손실로 체크포인트를 고르고, 블렌드 가중치도 같은 OOF에서 탐색한다.
5. **주관 라벨.** Q2·Q3는 센서만으로 chance 근처에 머물렀다. 남은 여지는 피험자내 순위 신호에 있다.

## 참고

- 코드: https://github.com/tjehdgus/etri-2026-sleep-prediction
- 개발 기록: https://mynews76154.tistory.com/15
- 주요 모델·논문: LightGBM, CatBoost, SigLIP-Large, NMF, TabPFN / NSF Sleep Duration Recommendations (Hirshkowitz et al., 2015), On the use of cross-validation for time series predictor evaluation (Bergmeir & Benítez, 2012), Sigmoid Loss for Language Image Pre-Training (Zhai et al., 2023), Learning to Rank using Gradient Descent (Burges et al., 2005)
