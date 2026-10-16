---
title: "Kaggle Store Sales 수요 예측: 16일 이상 시차 이동통계 피처와 매장×제품군별 LightGBM"
description: "에콰도르 식료품 체인 54개 매장, 33개 제품군의 16일 판매량을 예측하는 Kaggle Store Sales 과제에서 휴일·이벤트 정제, 16일 이상 시차의 SMA·EWM·Lag 피처, 매장×제품군 조합별 LightGBM 1,729개 모델로 RMSLE 0.42749, 724팀 중 68위를 기록했다."
date: 2026-03-12
lang: ko
key: store-sales
tags: [Time Series, Demand Forecasting, Feature Engineering, LightGBM, XGBoost, SARIMAX, Kaggle]
---

## 초록

Kaggle 「Store Sales - Time Series Forecasting」에서 에콰도르 식료품 체인의 매장(54개)×제품군(33개)별 일 판매량을 2017-08-16부터 16일간 예측했다. 2024년 부트캠프 4인 팀 프로젝트였고, 나는 조장으로 프로젝트 총괄과 LightGBM 모델을 맡았다. 전처리로 개업 전 기간과 판매 이력이 전부 0인 조합을 제거하고, 이전·대체 공휴일과 지역 단위 휴일을 정리해 피처로 붙였다. 시계열 분해와 주파수 분석으로 주간·월간 주기를 확인한 뒤, 예측 구간 길이 16일 이상의 시차만 쓰는 SMA 96개, EWM 40개, Lag 8개 피처를 만들었다. 매장×제품군 조합마다 log1p 타깃의 LightGBM을 따로 학습해(1,729개 모델) Kaggle RMSLE 0.42749, 724팀 중 68위를 기록했고, 팀원이 맡은 XGBoost 제출은 RMSLE 3.47144(801팀 중 765위)였다.

## 1. 문제 정의

- 과제: 매장×제품군 조합별 일 판매량(`sales`) 예측
- 입력: 일별 판매 이력, 프로모션 품목 수(`onpromotion`), 매장 정보(도시, 주, 유형, 클러스터), 휴일·이벤트, 유가, 매장별 거래 건수
- 출력: 2017-08-16 ~ 08-31, 16일 × 1,782개 조합 = 28,512개 예측값 (`test.csv` 행 수)
- 평가 지표: RMSLE (낮을수록 좋음)

```text
RMSLE = sqrt( (1/n) * Σ (log(1 + p_i) - log(1 + a_i))^2 )
```

RMSLE는 로그 공간의 오차라서 판매 규모가 큰 GROCERY I과 판매량이 적은 제품군의 상대 오차가 비슷한 무게로 반영된다. 제약 조건으로 테스트 구간의 판매량은 주어지지 않으므로, 판매 이력에서 만든 피처는 마지막 학습일(2017-08-15) 이전 값만 참조해야 한다. 예측 구간이 16일이라 테스트 마지막 날(08-31)에서 볼 수 있는 가장 최근 판매량은 16일 전 값이다.

## 2. 데이터

| 파일 | 내용 |
|---|---|
| `train.csv` | 2013-01-01 ~ 2017-08-15 일별 판매량, `onpromotion`. 압축 해제 기준 121.8MB, 3,000,888행(test `id`가 3000888부터 시작) |
| `test.csv` | 2017-08-16 ~ 08-31, 28,512행 |
| `stores.csv` | 54개 매장의 city, state, type, cluster |
| `holidays_events.csv` | 국가·지역·로컬 휴일, 이벤트, 이전(transferred) 여부 |
| `oil.csv` / `oil_updated.csv` | 일별 WTI 유가 1,218일. `oil_updated.csv`는 팀이 결측을 채운 버전 |
| `transactions.csv` | 매장별 일 거래 건수 |

EDA에서 확인한 사항은 다음과 같다(발표 자료 `docs/매장매출_시계열분석.pdf`).

- 매장마다 개업일이 달라 앞 구간이 0으로 채워져 있다.
- 매년 12월 25일은 모든 매장·제품군에서 행이 없다. 54 × 33 × 4년 = 7,128행 누락(`code/05_날짜 누락 확인 코드.py`).
- 판매량은 GROCERY I, PRODUCE, DAIRY 순으로 크고, 거래 건수는 주말에 급등한다.
- 2016년 4월 지진 시점 전후로 생필품 제품군은 증가, 비필수 제품군은 감소하는 패턴이 보인다.

### 2.1 전처리

| 단계 | 처리 | 근거 파일 |
|---|---|---|
| 개업 전 제거 | 8개 매장(52, 22, 42, 21, 29, 20, 53, 36)의 개업일 이전 행 삭제 | `code/02_개업전 가게 데이터 제거.py` |
| 미판매 조합 제거 | 학습 기간 판매량 합이 0인 (store, family) 조합을 anti join으로 제외 | `code/03`, `code/04` |
| 미판매 조합 예측 | 같은 조합의 테스트 16일은 0으로 고정 | `code/06_테스트 데이터 0채워넣기.py` |
| 12-25 누락 | 확인만 하고 채우지 않음. 트리 모델의 결측 처리에 맡김 | `code/05` |
| 휴일 정리 | 이전된 Holiday를 Transfer 날짜로 옮기고, Additional·Bridge를 Holiday로 통합, Work Day 분리 | `code/07` |

`code/04_판매량 0인 가게_family 제외(Anti Join).py`

```python
# Anti Join - 판매량이 쭉 0인 store+family 조합 제외
outer_join = train.merge(c[c.sales == 0].drop("sales", axis=1), how='outer', indicator=True)
train = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis=1)
del outer_join
gc.collect()
```

제거된 조합은 학습 데이터가 없어 모델 루프에서 건너뛰고, 제출 단계에서 0으로 채워진다. 예측 파일 `real4.csv`는 27,664행이고 test 28,512행과의 차이 848행(= 53개 조합 × 16일)이 0 고정분이다.

### 2.2 피처

| 범주 | 피처 | 수 |
|---|---|---|
| 매장 | `store_nbr`, `city`, `state`, `type`, `cluster` (범주형은 LabelEncoder) | 5 |
| 프로모션 | `onpromotion` | 1 |
| 휴일·이벤트 | 국가/지역/로컬 휴일 one-hot과 이진 플래그, `IsWorkDay`, 이벤트 one-hot(축구 경기는 `Futbol` 하나로 통합), 독립기념일·Cantonización·Fundación 플래그 | 가변 |
| 달력 | month, day_of_month, day_of_year, week_of_month, week_of_year, day_of_week, year, is_wknd, quarter, 월·분기·연 시작/끝 6개, season | 16 |
| 근무·급여일 | `workday`(휴일 또는 주말이면 0), `wageday`(15일과 월말, 공공부문 급여일) | 2 |
| SMA | 창 12종 × 시차 8종 | 96 |
| EWM | alpha 5종 × 시차 8종 | 40 |
| Lag | 시차 8종 | 8 |

지역 휴일은 `state`, 로컬 휴일은 `city`로 매장과 조인해 해당 지역 매장에만 휴일 플래그가 붙는다.

## 3. 방법

### 3.1 전체 파이프라인

```text
train / test / stores / holidays_events
  │
  ├─ [1] 로딩: dtype 축소 (sales float32, onpromotion float16, cluster int8)
  ├─ [2] 정제: 개업 전 행 삭제 → 미판매 조합 anti join → 해당 조합 테스트 0 고정
  ├─ [3] 휴일: transferred/Bridge/Additional 정리 → 국가·지역·로컬·이벤트 조인
  ├─ [4] 달력: 달력 16개 + workday, wageday
  ├─ [5] 시차 통계: (store, family)별 SMA 96 + EWM 40 + Lag 8, 최소 시차 16일
  │
  ├─ [6] 학습: 조합별 LightGBM (1,729개), 입력·타깃 log1p, 100 rounds
  ├─ [7] 예측: 08-16 ~ 08-31, expm1 역변환, 음수 → 0
  └─ [8] 제출: test와 (date, store_nbr, family)로 left join, 결측 0
```

[1]~[8]은 `code/01`~`code/14`에 단계별 스크립트로 남아 있다. 하나의 노트북 세션에서 순서대로 실행하는 구조라 각 파일은 앞 단계의 변수(`train`, `d`, `c`)를 그대로 쓴다. 예측 결과는 별도 Streamlit 대시보드(`github_dashboard.py`)에서 매장·제품군별 선 그래프로 확인했다.

### 3.2 주기 탐색

팀의 시계열 분석 결과(발표 자료)를 피처 설계의 기준으로 삼았다.

| 분석 | 결과 |
|---|---|
| 분해 강도 | 추세 65.6%, 계절성 24.4%, 잔차 6.3% |
| 월 단위 | 12월 매출이 가장 높고 8월이 가장 낮음 |
| 요일 단위 | 일요일 > 토요일 순으로 높고 목요일이 가장 낮음 |
| PSD(Welch), 푸리에 변환 | 7일, 3.5일 주기가 가장 강함. 30일, 15일도 유의. 84일, 94일, 187일, 421일 후보 |

주간 주기는 `day_of_week`, `is_wknd`, `workday`로, 반월·월 주기는 `wageday`, `day_of_month`, 시차 30일 통계로 반영했다. `wageday`는 코드 주석 기준 공공부문 급여일(15일, 월말)이며, PSD에서 확인한 15일 주기와 같은 간격이다.

### 3.3 시차 이동통계 피처

판매 이력 피처는 `(store_nbr, family)` 그룹 안에서 `shift(lag)` 후 집계한다. 시차 목록은 `[16, 30, 60, 90, 241, 23, 457, 169]`로 최소값이 예측 구간 길이 16이다. train과 test를 이어 붙인 `d`에서 계산하므로, 테스트 행은 판매량이 없는(NaN) 테스트 구간이 아니라 항상 2017-08-15 이전의 실제 판매량만 참조한다. 16일 미만 시차를 쓰면 테스트 후반부 행의 피처가 비게 된다.

`code/09_SMA features.py`

```python
def sma_features(dataframe, windows, lags):
    dataframe = dataframe.copy()
    for window in windows:
        for lag in lags:
            dataframe[f'SMA{window}_sales_lag{lag}'] = dataframe.groupby(["store_nbr", "family"])['sales'] \
                .transform(lambda x: x.shift(lag).rolling(window=window).mean())
    return dataframe

windows = [20, 30, 45, 60, 90, 120, 365, 730, 241, 23, 457, 169]
lags = [16, 30, 60, 90, 241, 23, 457, 169]
d = sma_features(d, windows, lags)
```

EWM은 같은 시차 목록에 alpha `[0.95, 0.9, 0.8, 0.7, 0.5]`를 곱해 40개, 단순 Lag는 8개다. SMA는 단기(20~60일)부터 1년(365), 2년(730) 창까지 두어 최근 수준과 연간 수준을 함께 준다. 발표 자료 기준으로 SMA·EWM·Lag 피처는 XGBoost 단계에서 먼저 만들었고 LightGBM에서 재사용했다.

### 3.4 조합별 LightGBM

매장마다, 그 안의 제품군마다 모델을 따로 학습한다. 54 × 33 = 1,782개 조합에서 미판매 53개를 뺀 1,729개 모델이고, 예측 행 수 27,664 = 1,729 × 16과 일치한다. 제품군 간 판매 규모와 주기 차이가 커서, 하나의 전역 모델에 `family`를 넣는 대신 조합 단위로 분리했다.

`code/13_LightGBM 모델 학습 및 예측.py`

```python
X_train = train_data.drop(columns=['sales', 'date', 'family']).values
y_train = train_data['sales'].values.reshape(-1, 1)

# 로그 정규화 적용
X_train_log = log_transform(X_train)
y_train_log = log_transform(y_train)

lgb_train = lgb.Dataset(X_train_log, y_train_log)
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'lambda_l2': 0.3,
    'verbose': -1
}
model = lgb.train(params, lgb_train, num_boost_round=100)
```

타깃에 `log1p`를 적용하고 L2 회귀(`objective: regression`)로 학습하므로, 학습 손실이 평가 지표 RMSLE의 제곱과 같은 형태가 된다. 예측은 `expm1`로 되돌리고 음수는 0으로 자른다. 입력 피처에도 `log1p`를 적용했는데, 단조 변환이라 트리 분할 결과에는 영향이 없다.

## 4. 실험

### 4.1 실험 설정

- 검증: 별도 홀드아웃이나 CV 없이 학습 기간 전체(2013-01-01 ~ 2017-08-14)로 학습하고 Kaggle 리더보드 점수로 비교했다.
- 환경(발표 자료): NVIDIA GTX 1080, Intel Core i5, CentOS 7 / Windows 10, Python 3.8, CUDA 11.7, cuDNN 8.5. LightGBM 코드는 GPU 옵션을 쓰지 않는다.

| 모델 | 담당 | 주요 하이퍼파라미터 | 근거 |
|---|---|---|---|
| LightGBM | 본인 | gbdt, objective regression, lambda_l2 0.3, num_boost_round 100, 나머지 기본값 | `code/13` |
| XGBoost | 팀원 | reg:squarederror, eta 0.1, max_depth 6, min_child_weight 3, subsample 0.8, colsample_bytree 0.8, lambda 1, alpha 0.1 | 발표 자료 |
| SARIMAX | 팀원 | ADF·KPSS 정상성 검정, AIC/BIC 기반 auto ARIMA, rolling forecasting | 발표 자료 |

### 4.2 결과

| 모델 | Kaggle RMSLE | 순위 |
|---|---|---|
| XGBoost | 3.47144 | 765 / 801 |
| **LightGBM (조합별, 최종)** | **0.42749** | **68 / 724** |

두 모델 모두 SMA·EWM·Lag 피처를 썼다. Getting Started 대회의 상시 리더보드라 제출 시점에 따라 참가 팀 수가 달라 순위의 분모가 다르다. XGBoost 학습 코드는 레포에 없어 두 점수 차이를 타깃 변환, 모델 단위 등으로 분리할 수는 없다.

### 4.3 비교 실험

**SARIMAX 개선 단계 (팀원 담당, 발표 자료)**

| 단계 | MAE | RMSE | R² |
|---|---|---|---|
| ARIMA 기준 | 165.41 | 612.53 | 0.795 |
| + 계절성(S) | 90.37 | 309.90 | 0.938 |
| + 데이터 전처리 | 74.76 | 251.93 | 0.957 |
| + Auto ARIMA | 57.29 | 132.43 | 0.96 |
| + Rolling forecasting | 51.58 | 136.42 | 0.96 |

계절성 항 추가가 MAE를 가장 크게 줄였다. 평가 대상 시계열과 검증 구간은 발표 자료에 적혀 있지 않아 Kaggle RMSLE와 직접 비교할 수 없다.

**모델 계열**

| 모델 | 결과 | 비고 |
|---|---|---|
| LightGBM | RMSLE 0.42749 | 최종 선택 |
| XGBoost | RMSLE 3.47144 | 같은 파생변수 사용 |
| ARIMA / SARIMAX | 위 표 | Kaggle 제출 점수 기록 없음 |
| LSTM, GRU | ARIMA/SARIMA보다 낮은 성과(README) | 수치 기록 없음 |
| Facebook Prophet | 시도 모델로만 기록 | 수치 기록 없음 |

## 5. 한계와 향후 과제

1. **검증 체계 부재.** 홀드아웃 없이 리더보드로만 비교했고, `num_boost_round=100`도 조기 종료 없이 고정했다. 2017-07-31 ~ 08-15 같은 마지막 16일을 검증 구간으로 두고 조합별 라운드 수를 정해야 한다.
2. **조합별 모델의 데이터 효율.** 1,729개 모델은 각자 조합당 최대 1,683일치 행으로 학습하고 다른 매장·제품군의 패턴을 공유하지 못한다. `family`, `store_nbr`를 범주형으로 넣은 전역 모델이나 제품군 단위 모델과 비교가 필요하다.
3. **쓰지 않은 외생 변수.** `oil_updated.csv`와 `transactions.csv`를 불러오지만 `d`에 조인하지 않아 유가와 거래 건수는 피처에 들어가지 않았다. 거래 건수는 테스트 구간 값이 없으므로 시차 피처로 넣어야 한다.
4. **사소한 구현 문제.** 학습 구간 조건이 `date < '2017-08-15'`라 마지막 학습일이 빠진다. `is_wknd = weekday // 4`는 금요일까지 주말로 표시한다. 시차 목록의 241, 457, 169일 같은 값은 주파수 분석 결과와의 대응이 기록되어 있지 않다.
5. **피처 기여 미측정.** SMA·EWM·Lag 144개 피처와 휴일 피처의 기여도를 따로 잰 실험이 없다. 피처 중요도와 그룹 단위 제거 실험으로 중복이 큰 창·시차를 줄일 수 있다.

## 참고

- 코드: https://github.com/tjehdgus/kaggle_lightgbm
- 대회: Kaggle, Store Sales - Time Series Forecasting
- 모델: LightGBM, XGBoost, ARIMA / SARIMAX
- 기법: Simple Moving Average, Exponentially Weighted Moving Average, 시계열 분해(추세·계절성·잔차), Welch Power Spectral Density, Fourier Transform
