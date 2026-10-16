---
title: "BARAM 2026 풍력발전량 예측: MOS 풍속 보정, FICR 인식 손실, BiGRU seq2seq 그룹별 블렌드"
description: "기상예보(LDAPS·GFS)만으로 가덕산 풍력단지 3개 그룹의 2025년 시간별 발전량을 예측했다. 대회 산식 채점 하네스, SCADA 기반 MOS 풍속 보정, 정산금 문턱을 넣은 손실, BiGRU seq2seq 중심의 그룹별 블렌드로 구성했다."
date: 2026-10-16
lang: ko
key: baram-wind
tags: [풍력발전량 예측, NWP, MOS, BiGRU, FICR, 시계열, 앙상블]
tistory: https://mynews76154.tistory.com/12
---

## 초록

데이콘 「제3회 풍력발전량 예측 AI 경진대회(BARAM 2026)」에서 가덕산 풍력단지(17기, 64MW) 3개 KPX 그룹의 2025년 시간별 발전량을 수치예보만으로 예측했다. 테스트 연도에는 발전량 이력이 없다. 대회 산식을 그대로 재현하는 검증 하네스를 먼저 만들었고, 오차 분해에서 그룹별 오차의 55~67%가 풍속 예보 오차로 나와 SCADA 실측 풍속을 타깃으로 하는 MOS 보정에 집중했다. 주력 모델은 하루 24시간을 한 샘플로 보는 3층 BiGRU seq2seq이고, 정산금 문턱(6%)을 시그모이드로 근사한 손실로 학습했다. 72시간 문맥 변형과 트리·표 모델을 그룹별로 블렌드해 2024년 홀드아웃 로컬 점수를 0.6525에서 0.6564로 올렸고, 로컬과 리더보드 점수의 상관은 약 0.91이었다.

## 1. 문제 정의

| 항목 | 내용 |
|---|---|
| 대상 | 3개 KPX 그룹, 설비용량 21.6 / 21.6 / 21.0MW |
| 출력 | 2025년 그룹별 시간별 발전량(kWh) |
| 입력 | LDAPS(1.5km, 16격자), GFS(25km, 9격자). SCADA는 학습 기간만 |
| 제약 | 기준 시점(전날 14시) 이후 공개 정보 금지, 원격 API 금지, 1일 5회 제출 |

```text
총점 = 0.5 * (1 - NMAE) + 0.5 * FICR      (각각 그룹별 계산 후 3그룹 평균)
NMAE = mean(|pred - y| / cap)
FICR = 획득 정산금 / 최대 정산금, 시간별 오차율 <=6%: 4원/kWh, <=8%: 3원/kWh, 초과: 0원
채점 대상 = 실제 발전량 >= 설비용량 10%인 시간
```

FICR은 오차율의 계단 함수이고 발전량으로 가중된다. MAE 감소와 정산금 증가가 같은 방향이 아니므로 손실과 블렌드 선택 모두 이 산식을 직접 기준으로 삼았다.

`src/features.py`
```python
def comp_score(y_true, y_pred, cap):
    m = y_true >= 0.1 * cap
    yt, yp = y_true[m], y_pred[m]
    err = np.abs(yp - yt) / cap
    nmae = err.mean()
    money = np.where(err <= 0.06, 4 * yt, np.where(err <= 0.08, 3 * yt, 0)).sum()
    ficr = money / (4 * yt).sum()
    return nmae, ficr, int(m.sum())
```

## 2. 데이터

| 데이터 | 내용 | 기간 |
|---|---|---|
| LDAPS | 10m·50m 바람, 2m 기온·습도, 기압, 경계층 고도, 강수 | 학습 2022~2024, 테스트 2025 |
| GFS | 10/80/100m·850/700hPa 바람, 돌풍, 850hPa 기온, 500hPa 지위고도, 전운량 등 | 같음 |
| SCADA | 10분 터빈별 풍속·출력. Vestas 12기(그룹 1·2), UNISON 5기(그룹 3) | 학습만 |
| 타깃 | KPX 계량 그룹별 시간 발전량. 그룹 3은 2022년 없음 | 학습만 |

피처는 `features.py`에서 약 317개를 만든다.

| 분류 | 내용 |
|---|---|
| 허브 높이 풍속 | 두 높이 풍속의 전단 지수로 117m 멱법칙 외삽, 격자 간 mean/max/min/std와 격자별 값 |
| 물리 | 습윤 공기밀도, `rho*ws³`, 8방위 섹터별 풍속, 850hPa-지상 기온차, LDAPS·GFS 불일치 |
| 시간 문맥 | 같은 예보 run 안의 diff, 3시간 이동 통계, 일 평균·최대, 24시간 FFT 에너지 |
| 도메인 | 난류 강도, 상하층 디커플링, 파워커브 구간(컷인 3, 정격 11m/s), 3시간 램프, 배열축과 풍향 정렬도 |

같은 예보일 값은 모두 기준 시점 전에 공개된 한 run에서 나오므로 run 안의 rolling·일 집계는 누수가 아니다.

## 3. 방법

### 3.1 전체 파이프라인

```text
LDAPS + GFS → features.py (~317 피처)
  └─ MOS v1 (LightGBM) / v2 (BiGRU): 예보 → SCADA 실측 풍속, OOF
     ├─ 파워커브·이론 출력·그룹 3 출력 제한 확률 피처
     └─ cv.py 하네스 (year / monthly, 대회 산식, 실험 로그)
          ├─ seq2seq BiGRU 3층, FICR 손실 λ 변형   ← 주력
          ├─ seq3d: D-2~D 72시간 문맥 + 연도 이전 게이트
          └─ MLP / LightGBM / XGBoost / ExtraTrees / TabNet / TabPFN v2 / pytabkit
               └─ 그룹별 0.1 간격 가중 그리드 → [0, cap] 클리핑 → 제출
```

- MOS: 보정 풍속을 모든 모델이 공유하는 피처로 붙인다.
- 하네스: 모델은 `fit_predict(X_tr, y_tr, X_va, group, cap)`만 구현하고 같은 폴드·채점·로그를 거친다.
- 블렌드: 검증 예측을 모아 그룹별로 총점 최대 가중치를 고른다.

### 3.2 오차 분해와 MOS 풍속 보정

`diagnose_errors.py`는 2024년 이전 데이터로 실측 풍속→발전량 isotonic 파워커브를 맞춘 뒤, MOS 보정 풍속을 넣은 예측과 SCADA 실측 풍속을 넣은 예측의 NMAE를 비교한다. 후자는 바람을 알아도 남는 변환 오차이고 둘의 차이가 풍속 예보 오차다. 그 비중이 그룹별 55~67%였다.

| 단계 | 모델 | OOF | 실측 풍속 상관 |
|---|---|---|---|
| v1 | LightGBM(MAE, 1,200 트리) | 6개 연속 블록 | 0.88 |
| v2 | BiGRU 2층, 하루 시퀀스, 3그룹 동시, 시드 2개 | 일 단위 6개 연속 블록 | 0.91 |

v2는 v1 컬럼을 입력에서 빼고 학습했다. 보정 풍속 위에 isotonic 파워커브 예상 발전량, `0.5*rho*A*v³`에 풍속 구간별 출력계수(베츠 한계 0.593 클리핑)를 곱한 이론 출력, 그룹 3 UNISON 터빈이 100kWh 계단에 고정되는 월·시각별 확률을 피처로 더했다.

### 3.3 FICR 인식 손실

용량 정규화 공간에서 오차가 6%를 넘으면 시그모이드 벌점이 붙고, 벌점은 실제 발전량에 비례한다. 채점 제외 시간은 가중치 0.1이다.

`src/model_seq.py`
```python
    def wloss(pred, y):
        mask = ~torch.isnan(y)
        y0 = torch.nan_to_num(y)
        w = torch.where(y0 >= 0.1, 1.0, W_INELIG) * mask
        err = torch.abs(pred - y0)
        # FICR-aware soft threshold penalty (lambda swept, 1.5 optimal):
        # pushes hours hovering near the 6% settlement threshold inside it
        loss = err + FICR_LAM * torch.sigmoid((err - 0.06) / 0.01) * y0
        return (loss * w).sum() / w.sum().clamp(min=1)
```

λ는 0.8, 1.5, 3.0, 5.0을 스윕해 1.5를 기본값으로 두었다. λ가 작으면 MAE형, 크면 정산금형이 되므로 0.8과 1.5 변형을 모두 블렌드에 넣었다. 결측 타깃은 마스크로 빠지므로 그룹 1·2의 2022년 데이터가 공유 인코더 학습에 쓰인다.

### 3.4 BiGRU seq2seq와 seq3d

전선 통과나 램프 타이밍은 앞뒤 시간을 함께 봐야 드러난다. 01~24시 예보를 `(일, 24, F)`로 묶고 출력은 `(24, 3그룹)`이다.

| 구성 | 값 |
|---|---|
| 구조 | Linear(F→128)+SiLU → BiGRU 3층(hidden 96, dropout 0.2) → Linear(192→64→3) |
| 학습 | AdamW(lr 1e-3, wd 1e-4), cosine, 최대 120 epoch, batch 32 |
| 조기 종료 | 학습 구간 마지막 10%(최소 30일), patience 15 |
| 평균 | 상위 3개 체크포인트 SWA, 시드 2~3개 평균 |

seq3d는 D-2, D-1, D일 예보 72시간을 입력으로 받아 마지막 24시간만 예측한다. 각 날짜 예보는 그 날짜 run에서 오고 모두 기준 시점 이전에 공개된다. 달력상 연속된 날만 창으로 묶는다.

`src/model_seq3d.py`
```python
    for i in range(CTX - 1, len(days)):
        if (days[i] - days[i - CTX + 1]).days == CTX - 1:
            ws.append(np.concatenate(xs[i - CTX + 1:i + 1], axis=0))
            keep.append(i)
```

입력이 긴 모델은 특정 연도에 과적합하기 쉬워, 한 해로만 학습하고 다음 해를 채점하는 연도 이전 검사(2022→2023, 2023→2024)의 최솟값을 채택 기준으로 두었다. 1일 seq에도 같은 프로토콜을 둬 직접 비교한다.

### 3.5 보조 멤버와 블렌드

| 멤버 | 설정 |
|---|---|
| LightGBM / XGBoost | MAE 계열 목적함수, GPU, 5,000~6,000 트리 |
| ExtraTrees | 400 트리, 2023년 대회 1위 노트북의 모델 계열 |
| MLP / TabNet | 시간 행 framing 신경망 |
| TabPFN v2 | 1만 행 서브샘플 3회 평균. 2.5/3은 비상업 라이선스라 제외 |
| pytabkit | RealMLP, TabM, FT-Transformer, TabR. 단독 점수와 seq 잔차 상관으로 후보 판단 |

그룹마다 0.1 간격 심플렉스를 전수 탐색해 2024년 총점 최대 가중치를 고르고 테스트 예측에 적용한다. seq3d 블렌드는 λ=0.8 seq, λ=1.5 seq, seq3d, MLP, XGBoost에 LightGBM 또는 FT-Transformer를 더한 6개 멤버이며, 기준 0.6525를 넘을 때만 제출 파일을 만든다.

## 4. 실험

### 4.1 실험 설정

| 항목 | 내용 |
|---|---|
| year | 2024 이전 학습, 2024 전체 검증 (이력 없는 1년 예측 구조) |
| monthly | 2023~2024 24개월을 `(연*12+월) % 4`로 나눈 4폴드 |
| 연도 이전 | 단년 학습 2022→2023, 2023→2024, 최솟값 기준 |
| 기록 | 총점·1-NMAE·FICR을 `experiments.csv`에 남기고 리더보드 값을 같은 행에 기입 |
| 노이즈 기준 | 재실행 변동 약 ±0.005, 이 안의 차이는 채택하지 않음 |
| 환경 | Python 3.12, PyTorch(CUDA 12.8), LightGBM·XGBoost GPU |

### 4.2 결과

| 항목 | 값 |
|---|---|
| 풍속 예보 오차 비중 (2024) | 그룹별 55~67% |
| MOS 실측 풍속 상관 | v1 0.88 → v2 0.91 |
| 로컬 vs 리더보드 | 상관 약 0.91, gap 약 +0.007 |
| Public (7/7 파이프라인 커밋 시점) | 0.64526 |
| 블렌드 로컬 (year) | 0.6525 → 0.6564 (seq3d, 연도 게이트, pytabkit 추가) |

### 4.3 비교 실험

| 변형 | 결과 | 판정 |
|---|---|---|
| MOS v2 시퀀스 보정 | 상관 0.88 → 0.91 | 채택 |
| FICR 인식 손실 | λ=1.5 기본, 0.8·1.5 블렌드 | 채택 |
| 채점 제외 시간 가중 0.25 → 0.1 | seq, GBM, XGB, pytabkit에 적용 | 채택 |
| seq3d + pytabkit 블렌드 | 0.6525 → 0.6564 | 채택 |
| 풍속³ ±3시간 래그 | 모델별 약 -0.002, 앙상블 -0.005 | 기각 |
| EWM/SMA 다중 스케일 평활 | 모든 모델 약 -0.005 | 기각 |
| Chronos-2, TimesFM 2.5 | 타깃 이력이 없어 재귀 예측이 필요, TimesFM은 외생 변수 미지원 | 기각 |
| 분위수 LightGBM FICR 후처리 | 기준 미달 | 기각 |
| 트리 변형(단조 제약, 잔차 부스팅, DART, 배깅, CatBoost) | 기준 미달 | 기각 |
| LightGBM 커스텀 FICR 목적함수 | 이득 없음 | 기각 |

## 5. 한계와 향후 과제

- `build_mos.py`의 isotonic 파워커브와 출력계수 테이블은 2024년을 포함한 전체 학습 기간으로 맞춘다. year 검증에 2024년 정답 일부가 물리 피처로 들어가 로컬 점수가 낙관적일 수 있다. 테스트 연도 정답은 쓰지 않았다.
- MOS OOF는 연속 블록 폴드라 미래로 과거를 예측하는 폴드가 섞여 있다. forward-chaining으로 바꿔야 한다.
- 블렌드 가중치를 2024년 한 해로 전수 탐색한다. 연도 이전 게이트를 블렌드 단계에도 적용해야 한다.
- MLP, ExtraTrees, TabNet은 채점 제외 시간 가중치가 0.25로 남아 다른 멤버(0.1)와 조건이 다르다.
- 연도 이전 검사 결과가 레포에 기록되지 않아 seq와 seq3d를 사후 비교할 수 없다.

## 참고

- 개발 기록(티스토리): https://mynews76154.tistory.com/12
- 모델: LightGBM, XGBoost, ExtraTrees, TabNet, TabPFN v2, RealMLP, TabM, FT-Transformer, TabR, Chronos-2, TimesFM 2.5
- 2023년 풍력발전량 예측 AI 경진대회 상위권 노트북(ExtraTrees + 특성 공학, FLAML, AutoGluon)
