---
title: "Kaggle Store Sales Forecasting: Lagged Rolling Statistics (16+ Days) and Per-Store-Family LightGBM"
description: "For Kaggle Store Sales, a 16-day forecast of daily sales across 54 stores and 33 product families of an Ecuadorian grocery chain, I cleaned holiday and event data, built SMA, EWM and lag features with lags of at least 16 days, and trained 1,729 LightGBM models, one per store-family pair, reaching RMSLE 0.42749 and rank 68 of 724."
date: 2026-03-12
lang: en
key: store-sales
tags: [Time Series, Demand Forecasting, Feature Engineering, LightGBM, XGBoost, SARIMAX, Kaggle]
---

## Abstract

On Kaggle "Store Sales - Time Series Forecasting", we forecast daily sales of an Ecuadorian grocery chain for 16 days starting 2017-08-16, for every store (54) × product family (33) pair. It was a four-person bootcamp team project in 2024; I was team lead and owned the LightGBM model. Preprocessing removed pre-opening periods and pairs whose sales were zero over the whole history, and turned transferred, bridge and regional holidays into features. After time-series decomposition and spectral analysis confirmed weekly and monthly cycles, we built 96 SMA, 40 EWM and 8 lag features, all with lags no shorter than the 16-day horizon. One LightGBM model per store-family pair (1,729 models) trained on a log1p target scored RMSLE 0.42749 on Kaggle, rank 68 of 724. The XGBoost submission, owned by a teammate, scored 3.47144 (rank 765 of 801).

## 1. Problem

- Task: predict daily `sales` for each store-family pair
- Inputs: daily sales history, number of promoted items (`onpromotion`), store metadata (city, state, type, cluster), holidays and events, oil price, per-store transaction counts
- Output: 2017-08-16 to 08-31, 16 days × 1,782 pairs = 28,512 predictions (row count of `test.csv`)
- Metric: RMSLE (lower is better)

```text
RMSLE = sqrt( (1/n) * Σ (log(1 + p_i) - log(1 + a_i))^2 )
```

Because the error is measured in log space, relative errors on a high-volume family such as GROCERY I and on low-volume families carry similar weight. Test-period sales are not given, so any feature derived from sales history may only look at values up to the last training day (2017-08-15). With a 16-day horizon, the most recent sales value visible from the last test day (08-31) is 16 days old.

## 2. Data

| File | Contents |
|---|---|
| `train.csv` | Daily sales and `onpromotion`, 2013-01-01 to 2017-08-15. 121.8 MB uncompressed, 3,000,888 rows (test `id` starts at 3000888) |
| `test.csv` | 2017-08-16 to 08-31, 28,512 rows |
| `stores.csv` | city, state, type, cluster for 54 stores |
| `holidays_events.csv` | National, regional and local holidays, events, `transferred` flag |
| `oil.csv` / `oil_updated.csv` | Daily WTI price over 1,218 dates; `oil_updated.csv` is the team's gap-filled version |
| `transactions.csv` | Daily transaction count per store |

EDA findings (slides, `docs/매장매출_시계열분석.pdf`):

- Stores opened on different dates, so early periods are padded with zeros.
- December 25 is missing for every store and family: 54 × 33 × 4 years = 7,128 missing rows (`code/05_날짜 누락 확인 코드.py`).
- Sales volume is highest for GROCERY I, PRODUCE and DAIRY; transaction counts spike on weekends.
- Around the April 2016 earthquake, essential families rose and non-essential ones fell.

### 2.1 Preprocessing

| Step | Operation | Source |
|---|---|---|
| Pre-opening removal | Drop rows before the opening date for 8 stores (52, 22, 42, 21, 29, 20, 53, 36) | `code/02_개업전 가게 데이터 제거.py` |
| Never-sold pairs | Exclude (store, family) pairs whose total training sales is 0, via anti join | `code/03`, `code/04` |
| Never-sold forecast | Fix the 16 test days of those pairs to 0 | `code/06_테스트 데이터 0채워넣기.py` |
| Missing 12-25 | Checked but left as is; trees handle the gaps | `code/05` |
| Holiday cleanup | Move transferred holidays to their Transfer date, merge Additional and Bridge into Holiday, separate Work Day | `code/07` |

`code/04_판매량 0인 가게_family 제외(Anti Join).py`

```python
# Anti Join - 판매량이 쭉 0인 store+family 조합 제외
outer_join = train.merge(c[c.sales == 0].drop("sales", axis=1), how='outer', indicator=True)
train = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis=1)
del outer_join
gc.collect()
```

Excluded pairs have no training rows, so the model loop skips them and the submission step fills them with 0. The prediction file `real4.csv` has 27,664 rows; the 848-row gap to the 28,512 test rows (53 pairs × 16 days) is exactly the zero-filled part.

### 2.2 Features

| Group | Features | Count |
|---|---|---|
| Store | `store_nbr`, `city`, `state`, `type`, `cluster` (categoricals label-encoded) | 5 |
| Promotion | `onpromotion` | 1 |
| Holidays and events | One-hot and binary flags for national/regional/local holidays, `IsWorkDay`, one-hot events (all football matches merged into `Futbol`), independence, Cantonización and Fundación flags | varies |
| Calendar | month, day_of_month, day_of_year, week_of_month, week_of_year, day_of_week, year, is_wknd, quarter, six month/quarter/year start/end flags, season | 16 |
| Work and pay days | `workday` (0 on holidays or weekends), `wageday` (15th and month end, public-sector pay days) | 2 |
| SMA | 12 windows × 8 lags | 96 |
| EWM | 5 alphas × 8 lags | 40 |
| Lag | 8 lags | 8 |

Regional holidays are joined on `state` and local holidays on `city`, so the flag only lands on stores in the affected area.

## 3. Method

### 3.1 Pipeline

```text
train / test / stores / holidays_events
  │
  ├─ [1] Load: downcast dtypes (sales float32, onpromotion float16, cluster int8)
  ├─ [2] Clean: drop pre-opening rows → anti-join never-sold pairs → fix their test rows to 0
  ├─ [3] Holidays: resolve transferred/Bridge/Additional → join national, regional, local, events
  ├─ [4] Calendar: 16 calendar features + workday, wageday
  ├─ [5] Lagged stats: per (store, family) SMA 96 + EWM 40 + Lag 8, minimum lag 16 days
  │
  ├─ [6] Train: one LightGBM per pair (1,729), log1p inputs and target, 100 rounds
  ├─ [7] Predict: 08-16 to 08-31, expm1 back-transform, clip negatives to 0
  └─ [8] Submit: left join to test on (date, store_nbr, family), fill missing with 0
```

Steps [1] to [8] live in `code/01` to `code/14` as step scripts. They were run in order in one notebook session, so each file reuses variables from the previous step (`train`, `d`, `c`). Predictions were inspected per store and family in a separate Streamlit dashboard (`github_dashboard.py`).

### 3.2 Cycle analysis

The team's time-series analysis (slides) set the reference for feature design.

| Analysis | Result |
|---|---|
| Decomposition strength | Trend 65.6%, seasonal 24.4%, residual 6.3% |
| Monthly | December highest, August lowest |
| Day of week | Sunday, then Saturday highest; Thursday lowest |
| PSD (Welch), Fourier transform | 7-day and 3.5-day cycles strongest; 30 and 15 days also significant; 84, 94, 187 and 421 days as candidates |

The weekly cycle enters through `day_of_week`, `is_wknd` and `workday`; the half-month and monthly cycles through `wageday`, `day_of_month` and the 30-day lag statistics. `wageday` marks public-sector pay days (15th and month end) according to the code comment, which matches the 15-day spacing seen in the PSD.

### 3.3 Lagged rolling statistics

Sales-history features are computed within each `(store_nbr, family)` group as `shift(lag)` followed by an aggregate. The lag list is `[16, 30, 60, 90, 241, 23, 457, 169]`, whose minimum equals the 16-day horizon. They are computed on `d`, the concatenation of train and test, so test rows always reference actual sales from 2017-08-15 or earlier, never the empty (NaN) test period. Any lag below 16 would leave features empty for the later test days.

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

EWM uses the same lags with alphas `[0.95, 0.9, 0.8, 0.7, 0.5]` (40 features); plain lags add 8 more. SMA windows run from short (20 to 60 days) to one year (365) and two years (730), giving both the recent and the annual level. According to the slides, the SMA, EWM and lag features were first built for the XGBoost model and then reused for LightGBM.

### 3.4 Per-pair LightGBM

A separate model is trained for every family within every store. Of 54 × 33 = 1,782 pairs, 53 never-sold pairs are excluded, leaving 1,729 models, consistent with 27,664 = 1,729 × 16 prediction rows. Families differ widely in volume and cycle shape, so the pipeline splits by pair instead of feeding `family` into one global model.

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

With a `log1p` target and an L2 objective (`objective: regression`), the training loss has the same form as the squared RMSLE. Predictions are mapped back with `expm1` and clipped at 0. `log1p` is also applied to the inputs; since it is monotonic, it does not change tree splits.

## 4. Experiments

### 4.1 Setup

- Validation: no hold-out or CV. Models were trained on the full training period (2013-01-01 to 2017-08-14) and compared by Kaggle leaderboard score.
- Environment (slides): NVIDIA GTX 1080, Intel Core i5, CentOS 7 / Windows 10, Python 3.8, CUDA 11.7, cuDNN 8.5. The LightGBM code does not enable GPU training.

| Model | Owner | Key hyperparameters | Source |
|---|---|---|---|
| LightGBM | me | gbdt, objective regression, lambda_l2 0.3, num_boost_round 100, other defaults | `code/13` |
| XGBoost | teammate | reg:squarederror, eta 0.1, max_depth 6, min_child_weight 3, subsample 0.8, colsample_bytree 0.8, lambda 1, alpha 0.1 | slides |
| SARIMAX | teammate | ADF and KPSS stationarity tests, AIC/BIC-based auto ARIMA, rolling forecasting | slides |

### 4.2 Results

| Model | Kaggle RMSLE | Rank |
|---|---|---|
| XGBoost | 3.47144 | 765 / 801 |
| **LightGBM (per pair, final)** | **0.42749** | **68 / 724** |

Both models used the SMA, EWM and lag features. The competition is a Getting Started event with a rolling leaderboard, so the team count differs by submission date. The XGBoost training code is not in the repo, so the gap cannot be attributed to target transform, model granularity or other factors.

### 4.3 Ablation

**SARIMAX improvement steps (teammate, slides)**

| Step | MAE | RMSE | R² |
|---|---|---|---|
| ARIMA baseline | 165.41 | 612.53 | 0.795 |
| + seasonality (S) | 90.37 | 309.90 | 0.938 |
| + preprocessing | 74.76 | 251.93 | 0.957 |
| + auto ARIMA | 57.29 | 132.43 | 0.96 |
| + rolling forecasting | 51.58 | 136.42 | 0.96 |

Adding the seasonal term gave the largest MAE drop. The slides do not state which series or validation window was evaluated, so these numbers are not comparable to the Kaggle RMSLE.

**Model families**

| Model | Result | Note |
|---|---|---|
| LightGBM | RMSLE 0.42749 | final choice |
| XGBoost | RMSLE 3.47144 | same derived features |
| ARIMA / SARIMAX | table above | no Kaggle score recorded |
| LSTM, GRU | below ARIMA/SARIMA (README) | no numbers recorded |
| Facebook Prophet | listed as attempted | no numbers recorded |

## 5. Limitations and future work

1. **No validation scheme.** Models were compared only on the leaderboard, and `num_boost_round=100` was fixed without early stopping. The last 16 days (2017-07-31 to 08-15) should serve as a validation window to tune rounds per pair.
2. **Data efficiency of per-pair models.** Each of the 1,729 models sees at most 1,683 daily rows and shares nothing with other stores or families. A global model with `family` and `store_nbr` as categoricals, or one model per family, should be compared.
3. **Unused exogenous data.** `oil_updated.csv` and `transactions.csv` are loaded but never joined into `d`, so oil price and transaction counts are not features. Transactions have no test-period values and would need to enter as lagged features.
4. **Minor implementation issues.** The training filter `date < '2017-08-15'` drops the last training day. `is_wknd = weekday // 4` also flags Fridays as weekend. How lags such as 241, 457 and 169 days map to the spectral findings is not recorded.
5. **Feature contribution not measured.** There is no experiment isolating the 144 SMA/EWM/lag features or the holiday features. Feature importance and group-wise removal could prune redundant windows and lags.

## References

- Code: https://github.com/tjehdgus/kaggle_lightgbm
- Competition: Kaggle, Store Sales - Time Series Forecasting
- Models: LightGBM, XGBoost, ARIMA / SARIMAX
- Techniques: Simple Moving Average, Exponentially Weighted Moving Average, time-series decomposition (trend, seasonal, residual), Welch Power Spectral Density, Fourier Transform
