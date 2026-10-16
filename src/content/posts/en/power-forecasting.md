---
title: "Hourly Power Forecasting for 100 Buildings: Domain Features and Per-Building GBDT Model Selection"
description: "For the Dacon 2025 Power Consumption Forecasting AI Competition, I built CDH, THI and per-building holiday features with pattern-matched spline imputation, then picked the best of XGBoost, LightGBM and CatBoost for each building by validation SMAPE, reaching a public LB SMAPE of 5.22."
date: 2026-10-16
lang: en
key: power-forecasting
tags: [Time Series, Energy Forecasting, Feature Engineering, XGBoost, LightGBM, CatBoost, Dacon]
tistory: https://mynews76154.tistory.com/18
---

## Abstract

In the Dacon 2025 Power Consumption Forecasting AI Competition (August 2025, two-person team), the task was to forecast hourly electricity consumption for 100 buildings over 2024-08-25 to 08-31, scored by SMAPE. Zero readings and manually flagged outliers were imputed with a cubic spline fitted over other weeks that share the same hour and holiday conditions. Features included cooling degree hours (CDH), a temperature-humidity index (THI), individual closing-day rules for 16 department-store buildings, and per-building hour-by-weekday consumption statistics. XGBoost, LightGBM and CatBoost were trained separately for each building and validated with a 12-fold, 7-day backward time-series split. Selecting, per building, the model with the lowest validation SMAPE (Comparison) gave the best public LB score of 5.22, against 5.29539 for the best single model (CatBoost) and 5.2633 for Ridge stacking. The final submission was a weighted average of leaderboard-selected candidates and placed 18th of 934 teams (reported).

## 1. Problem

- Task: hourly electricity consumption (kWh) forecasting for 100 buildings
- Input: hourly weather per building (temperature, rainfall, wind speed, humidity) and building metadata (type, gross floor area, cooled area, PV, ESS and PCS capacity)
- Output: 2024-08-25 00:00 to 08-31 23:00, 168 hours per building, 16,800 predictions in total
- Metric: SMAPE (lower is better)

```text
SMAPE = (100 / n) * Σ |F_t - A_t| / ((|A_t| + |F_t|) / 2)
```

Because SMAPE is a relative error, small buildings weigh as much as large ones, and a few hours with a large ratio error, such as predicting a closing day as a normal business day, cost a lot.

Constraints: sunshine and solar radiation are missing in the test period, so both were dropped from training as well. The training data covers a single summer (June to August), and the test week immediately follows it.

## 2. Data

| File | Content |
|---|---|
| `train.csv` | 2024-06-01 to 08-24, hourly weather and consumption for 100 buildings |
| `building_info.csv` | Building type, floor area, cooled area, PV/ESS/PCS capacity |
| `test.csv` | 2024-08-25 to 08-31 weather (no sunshine or solar radiation) |
| `sample_submission.csv` | `building_datetime` ID and prediction |

The training period is 85 days, which gives 2,040 rows per building and 204,000 rows overall. There are 10 building types (16 department stores, 8 to 10 buildings for each other type). According to the EDA table in the presentation, mean consumption by type ranges from 10316.94 kWh (IDC) to 1106.31 kWh (apartments), roughly a 9x spread.

EDA showed very different load curves across buildings, so per-building modeling became the default. Irregular fluctuations inside a building's normal pattern were kept, since they could also appear in the test week. Only points that clearly broke from the trend were flagged as outliers and imputed.

### 2.1 Preprocessing

`building_info` was joined on the building number, and two kinds of correction were applied to consumption.

| Target | Definition | Matching condition |
|---|---|---|
| Flagged outliers | Building, date and hour listed by hand in `SPECIFIC_OUTLIERS`: 25 buildings, 62 timestamps | Holiday targets: holiday values at the same hour and weekday. Non-holiday targets: holiday values at the same hour |
| Zeros | `power_consumption < 1e-10` | Same hour, same weekday, same holiday flag |

Values from other weeks that satisfy the condition are collected, a natural cubic spline is fitted with "weeks since the start of training" on the x-axis, and the value is read at the target week. Since consumption trends upward over the summer, this follows the weekly trend better than a plain median. With fewer than four matching points, the code falls back to the median of the same hour, then the same weekday, then the whole building.

`code/final.py`

```python
if len(pattern_data) >= 4:
    try:
        base_date = building_data['date_time'].min()
        target_week = (target_datetime - base_date).days / 7.0
        ...
        cs = CubicSpline(pattern_weeks, pattern_values, bc_type='natural')
        interpolated_value = cs(target_week)
        interpolated_value = max(0, interpolated_value)
```

### 2.2 Features

Features built by `create_features()`:

| Group | Features | Definition |
|---|---|---|
| Cyclic time | `sin/cos_hour`, `sin/cos_date`, `sin/cos_month`, `sin/cos_dayofweek` | hour (/23), month+day/31 (/12), month (/12), weekday (/7) |
| Season | `summer_sin/cos` | Phase over one cycle from 2024-06-01 to 09-14 |
| Closing days | `holiday`, `week_of_month` | Per-building holiday rules, alternate-Sunday flag |
| Temperature stats | `day_max/min/mean_temperature`, `day_temperature_range`, `enhanced_avg_temp` | Per building per day; daily mean from 3-hourly readings |
| Composite indices | `CDH`, `THI`, `WCT` | Cooling degree hours, temperature-humidity index, wind chill |
| Weather flags | `weather`, `hot_humid_condition`, `hours_since_rain` | ±3 hours around rain, >26°C and >70% humidity, hours since last rain (capped at 168) |
| Load statistics | `day_hour_mean/std`, `holiday_hour_mean/std`, `hour_mean/std` | Per-building training-period mean and std by (hour, weekday), (hour, holiday) and hour |
| Building | `total_area`, `cooling_area`, `cooling_ratio`, `building_size_category` | Area, cooled-area ratio, three size bins at 1,000 and 10,000 m2 |
| Lag / rolling | `temp_rolling_3h_mean`, `temp_rolling_6h_max`, `temp_lag_1h/3h`, `temp_diff_1h`, `humidity_lag_1h`, etc. | Per-building rolling and lag values |
| Nonlinear | `temperature_squared`, `humidity_squared`, temperature x hour, temperature x humidity | Squares and interactions |

## 3. Method

### 3.1 Pipeline overview

```text
train.csv / test.csv / building_info.csv
  │
  ├─ [1] Load & join: rename columns, merge building info, PV/ESS flags
  ├─ [2] Closing-day rules: per-building holiday (16 store-specific rules + default rule)
  ├─ [3] Imputation: 62 flagged outliers → zeros, pattern-matched cubic spline (train only)
  ├─ [4] Features: cyclic time, season, temperature stats, CDH/THI/WCT, load stats, lag/rolling
  │
  ├─ [5] Per-building training (100 buildings × 3 models)
  │     ├─ XGBoost   12 folds × 5 seeds = 60-model average, under-prediction weighted MSE
  │     ├─ LightGBM  12 folds = 12-model average, two-stage reweighted retraining
  │     └─ CatBoost  12 folds × 5 seeds = 60-model average, log target, two-stage reweighting
  │
  ├─ [6] Comparison: per building, keep the model with the lowest validation SMAPE
  ├─ [7] Stacking (comparison only): three model predictions → Ridge meta-model
  └─ [8] Final submission: weighted average of LB-selected candidates
```

Steps [1] to [4] are in `code/final.py`. The training code for [5] to [8] is not in the repository, so their settings and scores are taken from the presentation (`docs/한국에너지공단.pdf`).

### 3.2 Cooling load indices: CDH and THI

CDH is the sum of temperature excess over 26°C across the last 12 hours. It separates two hours with the same temperature by how long the heat has lasted. With only about 12 weeks of data per building, the cumulative index was given directly instead of asking the trees to learn the effect from combinations of lags. THI combines temperature and humidity, and WCT combines temperature and wind speed.

`code/final.py`

```python
def calculate_cdh_sliding(xs):
    ys = []
    for i in range(len(xs)):
        if i < 11:
            ys.append(np.sum(np.maximum(0, xs[:(i+1)] - 26)))
        else:
            ys.append(np.sum(np.maximum(0, xs[(i-11):(i+1)] - 26)))
    return np.array(ys)
```

### 3.3 Per-building closing-day rules

Regular buildings treat weekends and public holidays (06-06, 06-07, 08-15, 08-16) as closed. Department stores close on different days, so closing days were read off each building's load plot, where consumption drops sharply, and encoded as rules. The code has individual rules for 16 buildings: every Sunday (18), alternate Sundays (27, 40, 59, 63), the 10th of each month plus specific Sundays (29), alternate Mondays (32), specific Mondays (19, 45, 54, 79, 95), alternate Sundays plus extra Mondays (74), and no closing days (34, 73, 88).

`code/final.py`

```python
elif building_num == 29:
    if day == 10:
        return True
    specific_holidays = ['2024-06-23', '2024-07-28', '2024-08-25']
    return date.strftime('%Y-%m-%d') in specific_holidays
...
else:
    # 일반 건물들: 주말 + 공휴일 적용
    official_holidays = ['2024-06-06', '2024-06-07', '2024-08-15', '2024-08-16']
    return (day_of_week >= 5) or (date.strftime('%Y-%m-%d') in official_holidays)
```

The rules also cover closing days inside the test week (Sunday 08-25, Monday 08-26, and so on). The same `holiday` flag is used as a matching condition during imputation and as a group key for `holiday_hour_mean/std`, so one set of rules feeds preprocessing, features and prediction.

### 3.4 Per-building baseline: load statistics

For each building, the training-period mean and standard deviation by (hour, weekday), (hour, holiday flag) and hour were attached to both train and test rows. This gives the model the building's usual load level as a baseline and lets the weather features explain the deviation from it.

### 3.5 Models and under-prediction correction

All three models were trained per building. The presentation reports a clear under-prediction tendency for XGBoost, and each model received a correction.

- XGBoost: weighted MSE that gives a 100x weight to under-predicted samples. SMAPE was also tracked per weekday.
- LightGBM, CatBoost: a first-stage model identifies under-prediction patterns, and a second stage retrains with sample weights (alpha=1).
- CatBoost: trained on `log(y + 1e-6)`. The presentation notes that it tended to over-predict without further correction.

### 3.6 Per-building model selection (Comparison)

For each building, all three models were trained, their validation SMAPE compared, and the test predictions of the lowest-scoring model kept. Examples from the presentation: building 1 went to CatBoost (10.10088), building 2 to LightGBM (4.995961), and buildings 98 and 99 to XGBoost (9.265551, 3.136065).

## 4. Experiments

### 4.1 Setup

- Validation: 12 folds cut backward in 7-day blocks from the end of the training period. One day on each side of the validation block was removed from training.
- Seeds: 2025, 42, 123, 777, 999 (XGBoost, CatBoost).
- Hardware: not recorded in the repository.

| Model | Main hyperparameters | Ensemble |
|---|---|---|
| XGBoost | lr 0.1, n_estimators 1000, max_depth 5, subsample 0.9, colsample_bytree 0.8, min_child_weight 6, reg_alpha 1, reg_lambda 1 | 12 folds × 5 seeds = 60 models |
| LightGBM | lr 0.05, n_estimators 2000, max_depth 5, num_leaves 50, subsample 0.9, colsample_bytree 0.8, min_child_samples 15, reg_alpha 1, reg_lambda 1 | 12 folds = 12 models |
| CatBoost | lr 0.1, iterations 1000, depth 5, subsample 0.9, colsample_bylevel 0.8, log target | 12 folds × 5 seeds = 60 models |
| Stacking meta | Ridge (base models: XGB 28, LGBM 27, CatBoost 32 features) | |

### 4.2 Results

Public LB SMAPE, from the ensemble-candidate table in the presentation.

| Configuration | Public LB SMAPE |
|---|---|
| CatBoost, single (per building) | 5.29539 |
| Ridge stacking | 5.2633 |
| **Comparison v2 (best model per building)** | **5.22** |

The final submission was a weighted average of candidates chosen by leaderboard score. The weights and the final private score are not recorded in the repository. The final rank was 18th of 934 teams (reported).

### 4.3 Ablation

**Per-building vs per-type models**

| Model | Per building (100 models) | Per type (10 models) | Delta |
|---|---|---|---|
| CatBoost | 5.29539 | 5.78100 | +0.486 |
| LightGBM | 5.39090 | 5.81680 | +0.426 |
| XGBoost | 5.39160 | 5.7983 | +0.407 |

Grouping by type was worse by more than 0.4 for all three models. 2,040 rows per building were enough for tree models, and buildings of the same type still differ in size and operation, so pooling them mixes different load curves.

**Ensembling strategies**

| Configuration | Public LB SMAPE | Note |
|---|---|---|
| CatBoost, single | 5.29539 | Reference |
| CatBoost alpha ensemble | 5.3765 | Worse than reference |
| Comparison v1 | 5.56768 | Worse than reference |
| Comparison v2 | 5.22 | Best single-model parameters, log transform |
| Ridge stacking | 5.2633 | 0.043 worse than v2 |

For the change from Comparison v1 to v2, the presentation only states "best-performing parameters from the single models" and "log(y + 1e-6) transform".

## 5. Limitations and future work

1. **Leakage in the load statistics.** `create_enhanced_power_statistics()` computes hour-by-weekday means and stds over the whole training period and attaches them to training rows. Validation targets leak into features, so validation SMAPE may have been optimistic. These statistics should be recomputed per fold from the training portion only.
2. **Variance of per-building selection.** Each building has only 168 validation hours per fold, so when the three models score close together the choice is close to noise. Selecting on the 12-fold mean and averaging when the gap is small would be safer, and the final blend weights should come from OOF predictions rather than the public LB.
3. **Mismatch between code and training setup.** In `final.py`, a missing comma after `'humidity_squared'` merges two literals into the nonexistent column `'humidity_squaredtemp_rolling_3h_mean'`, leaving 44 entries in `feature_cols`. This also differs from the per-model feature counts in the presentation (28/27/32), so training most likely used separate lists. Without the training code, steps [5] to [8] cannot be reproduced.
4. **Formula checks.** The THI in the code uses humidity instead of temperature in the `(9/5*humidity - 26)` term, and WCT takes wind speed in m/s without converting to km/h. Non-holiday outliers are also matched with `holiday == True`, so they are filled with holiday values. These should be checked against intent, corrected to the standard formulas, and re-evaluated.
5. **Extrapolation of tree models.** Consumption trends upward, and GBDTs cannot predict beyond the training range. Detrending with a linear model and fitting the residual with GBDT, and a comparison with Transformer-based forecasters, remain to be tried.

## References

- Code: https://github.com/tjehdgus/dacon_elec_catboost
- Development log (Tistory, Korean): https://mynews76154.tistory.com/18
- Models: XGBoost, LightGBM, CatBoost, Ridge Regression
- Techniques: Cubic Spline Interpolation, Cooling Degree Hours, Temperature-Humidity Index, Wind Chill Temperature
