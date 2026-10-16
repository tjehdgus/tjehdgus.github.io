---
title: "Predicting seven sleep indicators from lifelog sensors: per-target blending of nine sources with forward-chaining reweighting"
description: "One day of phone and watch sensor data was used to predict seven binary sleep indicators for the following night. Nine heterogeneous sources were blended per target, and the S1/S4 weights were re-selected on time-ordered folds, giving a private Log-Loss of 0.57849 (22nd place)."
date: 2026-07-02
lang: en
key: etri-sleep
tags: [Lifelog, Sleep Prediction, Ensemble, Time-series CV, Distribution Shift, SigLIP]
tistory: https://mynews76154.tistory.com/15
---

## Abstract

The 5th ETRI Human Understanding AI Paper Competition (June 2026) asked for seven binary indicators of a night's sleep, predicted from the previous day of 12 smartphone and smartwatch sensor streams. The data covered 10 subjects, 450 training nights and 250 test nights, with test nights concentrated later in time than training nights. I built nine sources (gradient-boosted trees, four sequence networks, frozen SigLIP embeddings, a sleep-window reconstruction, and a date-neighbour label model), combined them with per-target greedy blending, and re-selected the S1/S4 weights on date-ordered folds. For Q1/Q2, an unsupervised day-state (NMF) routing and a within-subject pairwise ranking signal were added as logit residuals. The blend reached an OOF Log-Loss of 0.58431; the final submission scored 0.5584 on the public and 0.57849 on the private leaderboard (22nd). The write-up was accepted at ICTC 2026, though I did not proceed to publication. The entry was submitted under a team name; the code and paper in the repository are my own work.

## 1. Problem

- **Input**: raw sensor records keyed by `(subject_id, lifelog_date)`. Watch heart rate, pedometer and light; phone charging, activity, ambient sound, BLE, GPS, light, screen, app usage and Wi-Fi.
- **Output**: probabilities for seven labels describing the night of `sleep_date`.

| Label | Meaning | Binarisation |
|---|---|---|
| Q1 / Q2 / Q3 | sleep quality / fatigue / stress (survey) | relative to the subject's long-term mean |
| S1 / S2 / S3 / S4 | TST / sleep efficiency / sleep onset latency / WASO | whether the NSF guideline is met |

- **Metric**: mean of the per-label log-loss. Predicting 0.5 everywhere gives 0.693.

```text
L = (1/7) * sum_j [ -(1/N) * sum_i ( y_ij*log(p_ij) + (1-y_ij)*log(1-p_ij) ) ]
```

- **Constraints**: the same subjects appear in train and test, separated only by date. S1-S4 come from a Withings sleep sensor that is not among the inputs. Three submissions per day. All probabilities were clipped to `[0.02, 0.98]`.

## 2. Data

| Step | Details |
|---|---|
| Heart rate | each timestamp stores a per-second array; values outside 30-220 bpm dropped, then mean/min/max/sd |
| List-valued sensors | 8 ambient-sound class probabilities, GPS mean speed and lat/lon range, Wi-Fi/BLE device counts, total app time |
| Time windows | day-keyed all / day (9-18) / eve (18-24) / prebed (21-24); night-keyed night (0-7) / morning (6-10) / deep (0-4). Records from 0-10h are assigned to the previous night |
| Statistics | mean/std/min/max/median, plus sum for cumulative variables |
| Within-subject derivations | per column: lag-1 difference, 3-day rolling mean, EMA (span 3), expanding mean |
| Calendar | day of week, month, weekend flag, subject number |

1,015 base aggregates, four derivations each, and four calendar variables give 5,079 features (notebook output `FC5079`). The derivations encode "today relative to this person's usual", which matches the relative definition of the Q labels.

## 3. Method

### 3.1 Pipeline overview

```text
12 sensor parquet files → aggregation + within-subject derivations (5,079 features)
  → 9 base sources (OOF / test)
      ├─ tree        LGBM + CatBoost + ExtraTrees, top-150 by importance, pseudo-labels
      ├─ prior       per-subject label mean
      ├─ GRU / BiLSTM / TCN / Attn   CNN + sequence encoder + subject embedding, 8-seed average
      ├─ SigLIP      sensor image → frozen SigLIP-Large → LGBM
      ├─ SleepTree2  sleep-window reconstruction features + tree
      └─ Neighbor    exponentially weighted mean of date-adjacent labels
  → per-target greedy blend (Q1/Q2 add a day-state source)
  → S1/S4 weights re-selected on forward folds
  → Q1/Q2 logit residuals: day-state (α=0.45) → pairwise (α=0.15)
```

Sources were chosen for differing error patterns rather than standalone accuracy.

### 3.2 Base sources

| Source | Configuration |
|---|---|
| tree | top 150 features by ExtraTrees importance, selected inside each fold. `p = 0.4*CatBoost + 0.3*LGBM + 0.3*ET`. LGBM: num_leaves 7, min_child_samples 40, reg_lambda 5; CatBoost: depth 4, l2 8. Test rows with `max(p,1-p) >= 0.85` in a first pass were added as hard labels and the model retrained |
| prior | per-subject label mean on the fold's training part; starting point of the blend search |
| 4 DL models | 6 channels (hr, step, screen, charge, light, activity) × 24 hours. GRU, BiLSTM, TCN with dilations 1/2/4, 4-head attention. 8-dim subject embedding, 7-head BCE, AdamW lr 2e-3, dropout 0.4, 250 epochs. Predictions averaged over seeds 0-7 (DLens) |
| SigLIP | 12 channels × 24 hours resized to 256×256 and fed to `google/siglip-large-patch16-256`; weights frozen, pooler output fed to LGBM |
| SleepTree2 | the 15 reconstruction features below appended to the 5,079 features, trained like tree |
| Neighbor | `p_ij = sum_m w_im*y_mj / sum_m w_im`, `w_im = exp(-|date gap| / 3 days)`, over the same subject's training rows; OOF excludes the row itself |

**Sleep-window reconstruction.** This source approximates the quantities that S1-S4 threshold. On a 1,080-minute grid from 18:00 to 12:00 the next day:

```text
worn   = minutes with interpolated heart rate
rest   = (step <= 1) & worn, 11-min rolling mean > 0.5 → longest run (>= 30 min) is the main sleep candidate
asleep = inside that run & HR < median(HR_worn) + 5, 7-min rolling mean > 0.4
TST = asleep minutes, SE = TST/TIB, SOL = onset - bed start (earlier of run start and first charging)
WASO = (wake - onset + 1) - asleep minutes
flags: TST >= 420, SE >= 0.85, SOL <= 30, WASO <= 20
```

The thresholds are approximations set in code. Reconstruction v1 (median TST 256 min) and v3 (219 min) were too strict, so v2 was used.

**Why Neighbor.** Labels were autocorrelated over dates (Q1 +0.27, Q2 +0.30, S +0.15 to 0.24), and test nights are interleaved in time with training nights, so nearby labels carry a signal orthogonal to the sensors.

### 3.3 Fold design

Base-source OOF predictions used five folds (`folds_sh`): each subject's date-ordered rows were split into 10 blocks, and fold f held out blocks f and f+5. Every subject has one early and one late block hidden per fold, which leaks less between adjacent days than a random shuffle. The blend used random 5-fold splits and the forward folds below.

`solution_full.py`
```python
rs=allidx[np.argsort(sd)]                                   # 날짜순 정렬
# ★forward-chaining CV: 앞 날짜로 학습→뒤 날짜로 검증(test의 시간이동을 모사). 시작점 0.5~0.8을 미끄러뜨리며 4폴드
forward_folds=[(rs[:int(ntr*f)],rs[int(ntr*f):int(ntr*min(f+0.2,1.0))]) for f in [0.5,0.6,0.7,0.8]]
```

### 3.4 Per-target greedy blend

For each target, the search looks for non-negative weights summing to one. Starting from prior=1, it tries ±0.1 and ±0.05 on each source weight, keeps the single move with the lowest log-loss, and repeats up to 40 times; each fold continues from the previous fold's weights.

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

Coarse steps and a prior anchor were used because continuous optimisation degraded badly on the leaderboard (4.3). All seven targets were blended with random-fold weights; S1 and S4 were then re-searched on the forward folds and their weights replaced.

### 3.5 Q1/Q2 corrections

- **Day-state**: all 700 days (train and test) were binned into 48 half-hour slots × 6 channels (ambient-sound entropy, GPS movement, activity, screen, Wi-Fi and BLE counts) and factorised with 8-component NMF. Fourteen descriptors (state occupancy, transition count, first/last active slot, evening run length, routine deviation, entropy) plus their within-subject z-scores fed an LGBM source. The representation is transductive and label-free.
- **Pairwise**: up to 400 (positive night, negative night) pairs per subject trained an L2 logistic regression (C=0.05) on feature differences. Scores were converted to a rank within the subject's training scores and combined as `sigmoid(0.7*logit(prior) + 0.3*logit(rank))`.

`solution_full.py`
```python
    base = C(_sig(_logit(blend[:, j]) + 0.45*(_logit(ds_p[:, j]) - _logit(blend[:, j]))))
    blend2[:, j] = C(_sig(_logit(base) + 0.15*(_logit(_pair_src[j]) - _logit(base))))
```

## 4. Experiments

### 4.1 Setup

- Source OOF predictions used the `folds_sh` 5 folds (Neighbor used leave-one-out). Blend OOF fitted weights on each random fold's training part and applied them to its validation part.
- During development, blends were compared with nested scores in which the weight search was kept separate from the outer folds.
- RTX 5070 Ti 16GB, torch 2.11. A full run took 597 s (notebook output).

### 4.2 Results

| Source | OOF | Source | OOF |
|---|---|---|---|
| prior | 0.6270 | TCN | 0.6153 |
| tree | 0.6145 | Attn | 0.6193 |
| SleepTree2 | 0.6150 | day-state | 0.6290 |
| GRU | 0.6238 | SigLIP | 0.6951 |
| BiLSTM | 0.6252 | **9-source blend** | **0.58431** |

Per-label OOF (`assets/result_srcll.png`):

| Source | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|---|---|
| tree | 0.61 | 0.69 | 0.70 | 0.54 | 0.57 | 0.53 | 0.66 |
| prior | 0.66 | 0.69 | 0.69 | 0.58 | 0.58 | 0.54 | 0.65 |
| GRU | 0.66 | 0.68 | 0.68 | 0.59 | 0.58 | 0.55 | 0.62 |
| SleepTree2 | 0.60 | 0.69 | 0.70 | 0.53 | 0.57 | 0.55 | 0.66 |
| Neighbor | 0.67 | 0.62 | 0.67 | 0.65 | 0.62 | 0.60 | 0.70 |

The best source differs by label (SleepTree2 for S1 and Q1, GRU for S4, Neighbor for Q2), and Q2/Q3 stay near chance. Final scores were 0.5584 public (about 26th) and 0.57849 private (22nd); the private winner scored 0.56758.

### 4.3 Ablation

S1/S4 weights before and after forward reweighting (approximate values read from `assets/result_forward.png`):

| Target | Folds | tree | SigLIP | SleepTree2 | GRU | TCN | Neighbor |
|---|---|---|---|---|---|---|---|
| S1 | random | 0.33 | 0.11 | 0.56 | 0 | 0 | 0 |
| S1 | forward | 0.42 | 0.18 | 0.41 | 0 | 0 | 0 |
| S4 | random | 0.05 | 0 | 0 | 0.90 | 0 | 0.04 |
| S4 | forward | 0 | 0.05 | 0 | 0.67 | 0.10 | 0.19 |

Local scores and public LB per variant (`LEARNINGS.md`, `개발과정_팀원용.md`):

| Variant | Local | LB |
|---|---|---|
| Simple grid blend | OOF 0.6119 | 0.6081 |
| scipy continuous optimisation, 9 sources | OOF 0.5992 | 0.6231 (worse) |
| Fixed-weight blend | OOF 0.6112 | 0.61055 |
| 8 sources (+ image CNNs), greedy | OOF 0.5950 | 0.5994 |
| 10 sources (+ ResNeXt101, SigLIP) | OOF 0.5936 | 0.6016 (worse) |
| 2 CNNs → SigLIP + SleepTree2 | nested 0.6019 | 0.5960 |
| + Neighbor | nested 0.6003 | |
| + day-state routing + pairwise | | 0.5596 |
| + S1/S4 forward reweighting (strength raised stepwise) | | 0.5588 → 0.5584 |
| Per-target backward elimination + TabPFN | same-fold selection 0.5917, leak-free 0.6590 | no gain |
| Per-subject TabPFN | nested 0.81 | no gain |
| NSF-threshold 1D calibration (S1) | 0.639 (SleepTree2 0.530) | blend worse |
| Night-time HRV | blend 0.6019 → 0.6024 | |
| Sleep reconstruction v3 | S mean 0.581 (v2 0.573) | |

The changes that carried over to the leaderboard were source diversity, corrections aligned with the Q label definition, DL seed averaging, and time-ordered reweighting. Candidates that improved OOF while moving far from the current best predictions mostly got worse on the leaderboard.

## 5. Limitations and future work

1. **Ceiling on S1-S4.** The Withings data behind these labels is absent; across reconstruction variants the S-label log-loss stayed in the 0.53-0.58 range.
2. **How forward reweighting was evaluated.** The search computes loss only on each fold's `tr` part and never uses `va`, so in practice it is a sequential refit on date-ordered prefixes (50% to 80%). Its effect was confirmed only by repeated public submissions; a nested forward evaluation scored on the validation blocks is still needed.
3. **Global date ordering.** Forward folds sort all rows by date regardless of subject, so subjects with different recording periods can be over-represented in the training or validation part. A per-subject temporal split would mirror the test structure more closely.
4. **Optimistic OOF.** DL checkpoints are selected on validation-fold loss, and blend weights are searched on the same OOF predictions.
5. **Subjective labels.** Q2/Q3 stayed near chance from sensors alone; the remaining headroom is in within-subject ranking signals.

## References

- Code: https://github.com/tjehdgus/etri-2026-sleep-prediction
- Development log (Tistory, Korean): https://mynews76154.tistory.com/15
- Main models and papers: LightGBM, CatBoost, SigLIP-Large, NMF, TabPFN / NSF Sleep Duration Recommendations (Hirshkowitz et al., 2015), On the use of cross-validation for time series predictor evaluation (Bergmeir & Benítez, 2012), Sigmoid Loss for Language Image Pre-Training (Zhai et al., 2023), Learning to Rank using Gradient Descent (Burges et al., 2005)
