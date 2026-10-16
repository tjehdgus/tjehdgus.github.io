---
title: "BARAM 2026 Wind Power Forecasting: MOS Wind Correction, a FICR-Aware Loss, and Per-Group Blending around a BiGRU seq2seq"
description: "Hourly 2025 generation for three groups of the Gadeoksan wind farm was forecast from NWP (LDAPS, GFS) alone. The pipeline combines a metric-exact scoring harness, SCADA-based MOS wind correction, a loss that encodes the settlement thresholds, and a per-group blend centered on a BiGRU seq2seq."
date: 2026-07-14
lang: en
key: baram-wind
tags: [Wind Power Forecasting, NWP, MOS, BiGRU, FICR, Time Series, Ensemble]
tistory: https://mynews76154.tistory.com/12
---

## Abstract

For the Dacon "3rd Wind Power Forecasting AI Competition (BARAM 2026)", I forecast hourly 2025 generation for the three KPX metering groups of the Gadeoksan wind farm (17 turbines, 64 MW) using numerical weather prediction only. No generation history is available for the test year. A validation harness that reproduces the competition metric exactly was built first. An error decomposition attributed 55-67% of each group's error to the wind-speed forecast, so effort went into MOS correction with SCADA-measured hub wind as the target. The main model is a 3-layer BiGRU seq2seq that treats one forecast day (24 hours) as a sample, trained with a loss that approximates the 6% settlement threshold with a sigmoid. Blending it with a 72-hour-context variant and tree/tabular members per group raised the 2024 holdout score from 0.6525 to 0.6564, and local scores tracked the leaderboard with a correlation of about 0.91.

## 1. Problem

| Item | Description |
|---|---|
| Target | 3 KPX groups, capacity 21.6 / 21.6 / 21.0 MW |
| Output | Hourly generation (kWh) per group for 2025 |
| Input | LDAPS (1.5 km, 16 grid points), GFS (25 km, 9 grid points); SCADA only for the training period |
| Constraints | No information published after the base time (14:00 KST of the previous day), no remote API models, 5 submissions per day |

```text
score = 0.5 * (1 - NMAE) + 0.5 * FICR      (each computed per group, then averaged over 3 groups)
NMAE  = mean(|pred - y| / cap)
FICR  = earned settlement / max settlement; hourly error <=6%: 4 KRW/kWh, <=8%: 3 KRW/kWh, else 0
scored hours = actual generation >= 10% of capacity
```

FICR is a step function of the hourly error rate, weighted by generation. Lowering MAE and earning more settlement do not always move together, so both the training loss and blend selection were tied directly to this metric.

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

## 2. Data

| Source | Content | Period |
|---|---|---|
| LDAPS | 10 m / 50 m wind, 2 m temperature and humidity, pressure, boundary-layer height, precipitation | train 2022-2024, test 2025 |
| GFS | 10/80/100 m and 850/700 hPa wind, gust, 850 hPa temperature, 500 hPa geopotential, cloud cover, etc. | same |
| SCADA | 10-min per-turbine wind speed and power; Vestas x12 (groups 1, 2), UNISON x5 (group 3) | train only |
| Target | Hourly KPX-metered generation per group; group 3 has no 2022 labels | train only |

`features.py` builds about 317 features.

| Category | Content |
|---|---|
| Hub-height wind | Shear exponent from two heights, power-law extrapolation to 117 m; cross-grid mean/max/min/std plus per-grid values |
| Physics | Moist air density, `rho*ws^3`, wind speed per 8 direction sectors, 850 hPa minus surface temperature, LDAPS-GFS disagreement |
| Temporal context | Within-run diff, 3-hour rolling stats, daily mean/max, 24-hour FFT energy |
| Domain | Turbulence intensity, upper/surface decoupling, power-curve zones (cut-in 3, rated 11 m/s), 3-hour ramp, alignment of wind with the turbine-row axis |

All values for one forecast day come from a single run published before the base time, so rolling windows and daily aggregates within a run do not leak.

## 3. Method

### 3.1 Pipeline

```text
LDAPS + GFS → features.py (~317 features)
  └─ MOS v1 (LightGBM) / v2 (BiGRU): forecast → SCADA-measured wind, OOF
     ├─ power-curve, theoretical-power and group-3 curtailment features
     └─ cv.py harness (year / monthly, competition metric, experiment log)
          ├─ seq2seq BiGRU x3, FICR loss λ variants   ← main model
          ├─ seq3d: 72-hour D-2..D context + year-transfer gate
          └─ MLP / LightGBM / XGBoost / ExtraTrees / TabNet / TabPFN v2 / pytabkit
               └─ per-group weight grid (step 0.1) → clip to [0, cap] → submission
```

- MOS: corrected wind speed is attached as a shared feature for every model.
- Harness: each model implements only `fit_predict(X_tr, y_tr, X_va, group, cap)` and goes through the same folds, scoring, and logging.
- Blend: validation predictions are collected and the score-maximizing weights are chosen per group.

### 3.2 Error decomposition and MOS wind correction

`diagnose_errors.py` fits an isotonic power curve (measured wind to generation) on pre-2024 data, then compares the NMAE of predictions fed with MOS-corrected wind against predictions fed with SCADA-measured wind. The latter is the conversion error that remains even with perfect wind; the gap between the two is the wind-forecast error. That share was 55-67% depending on the group.

| Stage | Model | OOF | Corr. with measured wind |
|---|---|---|---|
| v1 | LightGBM (MAE, 1,200 trees) | 6 contiguous blocks | 0.88 |
| v2 | 2-layer BiGRU over the day, 3 groups jointly, 2 seeds | 6 contiguous day blocks | 0.91 |

v2 was trained with the v1 columns removed from its inputs. On top of the corrected wind, three features were added: expected generation from the isotonic power curve, theoretical power `0.5*rho*A*v^3` multiplied by a per-wind-bin power coefficient (clipped at the Betz limit 0.593), and a month-by-hour probability that group 3's UNISON turbines sit on 100 kWh curtailment steps.

### 3.3 FICR-aware loss

In capacity-normalized units, a sigmoid penalty switches on when the error exceeds 6%, scaled by actual generation. Unscored hours get weight 0.1.

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

λ was swept over 0.8, 1.5, 3.0, and 5.0, and 1.5 became the default. A small λ yields an MAE-leaning model and a large one a settlement-leaning model, so both the 0.8 and 1.5 variants entered the blend. Missing targets are masked out, which lets 2022 data from groups 1 and 2 train the shared encoder even though group 3 has no 2022 labels.

### 3.4 BiGRU seq2seq and seq3d

Front passages and ramp timing only show up when neighboring hours are seen together. Forecast hours 01-24 were stacked into `(days, 24, F)` with a `(24, 3 groups)` output.

| Component | Value |
|---|---|
| Architecture | Linear(F→128)+SiLU → BiGRU x3 (hidden 96, dropout 0.2) → Linear(192→64→3) |
| Training | AdamW (lr 1e-3, wd 1e-4), cosine schedule, up to 120 epochs, batch 32 |
| Early stopping | Last 10% of training days (min 30), patience 15 |
| Averaging | SWA over the top 3 checkpoints, 2-3 seeds averaged |

seq3d takes 72 hours of D-2, D-1, and D forecasts and predicts only the last 24 hours. Each day's forecast comes from that day's run, all published before the base time. Only calendar-consecutive days form a window.

`src/model_seq3d.py`
```python
    for i in range(CTX - 1, len(days)):
        if (days[i] - days[i - CTX + 1]).days == CTX - 1:
            ws.append(np.concatenate(xs[i - CTX + 1:i + 1], axis=0))
            keep.append(i)
```

Longer-context models overfit to particular years more easily, so the acceptance criterion was the minimum score of a year-transfer test: train on a single year and score the next (2022→2023, 2023→2024). The same protocol exists for the 1-day seq model, so the two framings are directly comparable.

### 3.5 Auxiliary members and blending

| Member | Setup |
|---|---|
| LightGBM / XGBoost | MAE-type objectives, GPU, 5,000-6,000 trees |
| ExtraTrees | 400 trees; the model family of the 2023 edition's winning notebook |
| MLP / TabNet | Neural nets on the hourly-row framing |
| TabPFN v2 | Mean of 3 subsamples of 10k rows; v2.5/v3 excluded due to non-commercial licenses |
| pytabkit | RealMLP, TabM, FT-Transformer, TabR; screened by standalone score and residual correlation with seq |

For each group, a 0.1-step simplex was searched exhaustively for the weights maximizing the 2024 score, and the weights were applied to the test predictions. The seq3d blend uses six members (λ=0.8 seq, λ=1.5 seq, seq3d, MLP, XGBoost, plus LightGBM or FT-Transformer) and writes a submission file only when it beats the 0.6525 baseline.

## 4. Experiments

### 4.1 Setup

| Item | Description |
|---|---|
| year | Train before 2024, validate on all of 2024 (mirrors forecasting a full year with no history) |
| monthly | 24 months of 2023-2024 split into 4 folds by `(year*12+month) % 4` |
| Year transfer | Single-year training 2022→2023, 2023→2024; minimum is the criterion |
| Logging | Score, 1-NMAE, and FICR written to `experiments.csv`; leaderboard values filled into the same row |
| Noise threshold | Re-runs vary by about ±0.005; differences inside this band were not adopted |
| Environment | Python 3.12, PyTorch (CUDA 12.8), GPU LightGBM and XGBoost |

### 4.2 Results

| Item | Value |
|---|---|
| Wind-forecast share of error (2024) | 55-67% per group |
| MOS correlation with measured wind | v1 0.88 → v2 0.91 |
| Local vs leaderboard | correlation ~0.91, gap ~+0.007 |
| Public score (Jul 7 pipeline commit) | 0.64526 |
| Blend, local (year) | 0.6525 → 0.6564 (after adding seq3d, transfer gate, pytabkit members) |

### 4.3 Ablation

| Variant | Result | Decision |
|---|---|---|
| MOS v2 sequence correction | corr. 0.88 → 0.91 | adopted |
| FICR-aware loss | λ=1.5 default, 0.8 and 1.5 both blended | adopted |
| Unscored-hour weight 0.25 → 0.1 | applied to seq, GBM, XGB, pytabkit | adopted |
| seq3d + pytabkit blend | 0.6525 → 0.6564 | adopted |
| ±3 h lags of ws^3 | about -0.002 per model, -0.005 for the ensemble | rejected |
| Multi-scale EWM/SMA smoothing | about -0.005 for every model | rejected |
| Chronos-2, TimesFM 2.5 | no target history, so forecasting must be recursive; TimesFM takes no covariates | rejected |
| Quantile-LightGBM FICR post-processing | below threshold | rejected |
| Tree variants (monotone, residual boosting, DART, bagging, CatBoost) | below threshold | rejected |
| LightGBM custom FICR objective | no gain | rejected |

## 5. Limitations and future work

- The isotonic power curve and power-coefficient table in `build_mos.py` are fit on the whole training period, including 2024. Some 2024 labels therefore reach the year validation through the physics features, which may make local scores optimistic. Test-year labels were never used.
- MOS OOF uses contiguous block folds, so some folds predict the past from the future. Forward-chaining folds would match the test setting.
- Blend weights are searched exhaustively against a single year (2024). The year-transfer gate should also be applied at the blend stage.
- MLP, ExtraTrees, and TabNet still use 0.25 for unscored hours, unlike the other members (0.1).
- Year-transfer results were not logged in the repository, so seq and seq3d cannot be compared after the fact.

## References

- Development log (Tistory, Korean): https://mynews76154.tistory.com/12
- Models: LightGBM, XGBoost, ExtraTrees, TabNet, TabPFN v2, RealMLP, TabM, FT-Transformer, TabR, Chronos-2, TimesFM 2.5
- Top notebooks from the 2023 Wind Power Forecasting AI Competition (ExtraTrees + feature engineering, FLAML, AutoGluon)
