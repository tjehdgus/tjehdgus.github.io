---
title: "SPI6 Drought-Stage Classification: Joining Weather and Dam Daily Data, Moving from Oracle to R Random Forest, and an R Shiny Probability Dashboard"
description: "Joined KMA ASOS/SPI6 data with K-water daily dam records at the station-day level, trained a random forest that classifies six SPI6-based drought stages, and served per-stage probabilities through an R Shiny dashboard. Honorable mention, 2024 K-water Public Water Big Data Contest."
date: 2026-03-10
lang: en
key: kwater-drought
tags: [Random Forest, SPI6, Drought Classification, Oracle Data Mining, R Shiny, Imbalanced Classification]
tistory: https://mynews76154.tistory.com/20
---

## Abstract

This project framed drought monitoring as a classification task: the 6-month Standardized Precipitation Index (SPI6) was binned into six drought stages, and a model predicted the stage from daily weather and hydrological variables. KMA ASOS daily observations, station-level SPI6, a cumulative precipitation feature derived from daily rainfall, and K-water multipurpose/water-supply dam records were joined in Oracle on station and date. The first models were trained inside the database with Oracle `DBMS_DATA_MINING` (`ALGO_RANDOM_FOREST`) to compare periods and feature sets; once per-class metrics and hyperparameter search were needed, the work moved to R `randomForest` and `caret`. Twelve candidate features were reduced to five based on information gain and whether a user could supply the value at prediction time, with per-feature min-max or z-score scaling. The final model as presented reached 96.1% accuracy, F1 0.933, and AUC 0.955 on 47,098 rows (42,392 train / 4,706 test), and was deployed in an R Shiny dashboard that shows the probability of each drought stage. The entry received an honorable mention.

## 1. Problem

- Task: given the daily weather and hydrological state of a dam/station, classify its SPI6 drought stage and show per-stage probabilities in a dashboard.
- Input: cumulative precipitation, possible sunshine duration (가조시간), mean temperature, reservoir storage, and effective storage capacity (final five features).
- Output: one of six classes plus class probabilities (`predict(..., type = 'prob')`).
- Target: continuous SPI6 was binned with SQL `UPDATE` statements.

| Stage | SPI6 range |
|---|---|
| Severely wet | `spi6 >= 1.50` |
| Moderately wet | `1.00 <= spi6 < 1.50` |
| Normal | `-1.00 < spi6 < 1.00` |
| Mild drought | `-1.50 < spi6 <= -1.00` |
| Moderate drought | `-2.00 < spi6 <= -1.50` |
| Severe drought | `spi6 <= -2.00` |

`code/sql/03_랜포_14-23.sql` tried a five-stage version that merged the two wet stages, but the final dataset and the Shiny app use six stages.

- Metrics: the Oracle stage reported test accuracy only. In R, because of class imbalance (Normal 68.2%, Severe drought 1.9%), per-class F1 and macro/micro-averaged F1 from the confusion matrix became the primary metrics.

```text
precision_c = TP_c / (TP_c + FP_c)
recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)
macro_F1    = mean_c(F1_c)
```

  Secondary metrics were sensitivity, specificity, and precision from `caret::confusionMatrix` and multiclass AUC from `pROC::multiclass.roc` (`code/R/06_SPI_성능평가_F1_ROC.R`).
- Constraint: only values a dashboard user can enter are allowed as inputs. Quantities determined by dam operation, such as outflow, are excluded.

## 2. Data

### 2.1 Sources

| Table (repo CSV) | Source | Period | Granularity | Main columns |
|---|---|---|---|---|
| `기상_랜포_사용_테이블.csv` | KMA ASOS daily | 2014-01-01 to 2023-12-31 | 92 stations x day | mean temperature, possible sunshine duration, total sunshine hours, mean relative humidity, mean ground temperature, etc. |
| `SPI_랜포_사용_테이블.csv` | KMA SPI6 | 2014-01-01 to 2023-12-31 | 62 stations x day | SPI6, season, province, drought stage |
| `누적강수_랜포_사용_테이블.csv` | derived from KMA daily rainfall | 2013-07-01 to 2023-12-31 | 89 stations x day | daily rainfall, cumulative precipitation, river basin (Han, Geum, Nakdong, Yeongsan-Seomjin) |
| `18-22_댐_랜포_테이블.csv` | K-water dam daily records | 2018-01-01 to 2022-12-31 | 35 dams x day | storage, effective storage, total storage, outflow, rainfall |
| `일별_지역별_평균_지하수수위.csv` | groundwater level | 2018-01-01 to 2022-12-31 | 119 stations x day | level |

Checked against the repo data, the cumulative precipitation column matches a roughly 6-month (182 to 184 day) rolling sum of daily rainfall, which is also why that table starts in July 2013.

### 2.2 Preprocessing

- Joins: weather, SPI, and cumulative precipitation were joined on `(date, station)`, and dam records on `(date, station)` via the dam's mapped weather station. `05_랜포_지하수수위_저수량_방류량.sql` built `weather_p` and `accper_p` with `substr(지점명, 1, 2)` to normalize station names and reduce join misses.
- Missing values: rows were dropped column by column with `delete from w where <col> is null`. The initial all-features version (`01_테이블생성.sql`) did this for some 50 ASOS columns.
- Station mapping: `code/etc/기상청_평균온도.txt` maps stations without data to neighboring stations (e.g., Sejong to Cheongju, Cheonan, Daejeon).
- Groundwater: `code/etc/지하수수위_코드.txt` aggregates per-station mean levels with `round(avg(평균수위), 1)` into yearly tables and merges them with `union all`.

### 2.3 Modeling datasets

| CSV | Rows | Columns | Note |
|---|---|---|---|
| `SPI_랜포_테이블_최종.csv` | 32,239 | stage, SPI6, 12 candidate features, groundwater level | before, 18 dams |
| `SPI_예측테이블_최종.csv` | 47,098 | stage, cumulative precipitation, sunshine duration, mean temperature, storage, dam, total storage | after, 26 dams |

Class distribution (counted from the repo CSVs):

| Stage | Before (32,239) | After (47,098) |
|---|---|---|
| Normal | 21,999 (68.2%) | 32,017 (68.0%) |
| Mild drought | 3,285 (10.2%) | 4,531 (9.6%) |
| Moderately wet | 3,176 (9.9%) | 4,858 (10.3%) |
| Moderate drought | 1,635 (5.1%) | 2,388 (5.1%) |
| Severely wet | 1,537 (4.8%) | 2,475 (5.3%) |
| Severe drought | 607 (1.9%) | 829 (1.8%) |

Dropping groundwater level recovered station-days with no groundwater observation, which grew the dataset from 32,239 to 47,098 rows.

## 3. Method

### 3.1 Pipeline

```text
[KMA ASOS daily]  [KMA SPI6]  [daily rainfall -> 6-month cumulative]  [K-water dam daily]  [groundwater]
        └─────────────┴──────────────────┬───────────────┴─────────────────────┘
                                         ▼
                 Oracle: normalize station names -> join on (date, station) -> drop NULL rows
                                         ▼
                 Oracle: SPI6 -> drought stage (UPDATE, 6 stages)
                                         ▼
                 Oracle DBMS_DATA_MINING: ALGO_RANDOM_FOREST, accuracy comparison
                 (periods 19-22 / 14-23 / 18-22; add storage, province, groundwater, outflow)
                                         ▼
                 CSV export -> R
                 ├─ feature selection: 12 -> 5 (information gain + user-suppliable)
                 ├─ scaling: min-max (sunshine, temperature), z-score (storage, cumulative precip)
                 ├─ 9:1 stratified split (createDataPartition)
                 ├─ ntree sweep 1..100, macro F1
                 └─ evaluation: per-class F1, sensitivity, specificity, precision, multiclass AUC
                                         ▼
                 R Shiny: pick basin/dam -> scale inputs with train stats -> per-stage probability bars
```

Integration, labeling, and the first model comparisons stayed in Oracle; the work moved to R once richer metrics and hyperparameter search were needed. The dashboard retrains the model on the same data at app startup.

### 3.2 Labeling in SQL

The stage column was added to the SPI6 table with `alter table ... add` and filled with one `UPDATE` per range.

`code/sql/02_랜포_19-22.sql`

```sql
alter table drou add 가뭄 varchar2(20);
update drou
    set 가뭄 = '심한습윤'
    where spi6 >= 1.50;
update drou
    set 가뭄 = '보통습윤'
    where spi6 >= 1.00 and spi6 < 1.50;
update drou
    set 가뭄 = '정상'
    where spi6 > -1.00 and spi6 < 1.00;
update drou
    set 가뭄 = '약한가뭄'
    where spi6 > -1.50 and spi6 <= -1.00;
update drou
    set 가뭄 = '보통가뭄'
    where spi6 > -2.00 and spi6 <= -1.50;
update drou
    set 가뭄 = '심한가뭄'
    where spi6 <= -2.00;
commit;
```

Only the stage label went into the model tables, never `spi6` itself. In R (`07_랜포_트리수별_F1_비교.R`), `spi[, -c(1, 2, 4, 8, 9, 10, 13, 14)]` also removes the SPI6 column so the raw target value cannot leak into the inputs.

### 3.3 Random forest inside Oracle

Since the data already lived in Oracle, the first model was trained in the database. A settings table copied from `DBMS_DATA_MINING.GET_DEFAULT_SETTINGS` received the algorithm name and the automatic data preparation flag.

`code/sql/04_랜포_저수량포함_최종.sql`

```sql
BEGIN
   INSERT INTO SETTINGS_GLM
        VALUES (DBMS_DATA_MINING.ALGO_NAME, 'ALGO_RANDOM_FOREST');

   INSERT INTO SETTINGS_GLM
        VALUES (DBMS_DATA_MINING.PREP_AUTO, 'ON');

   INSERT INTO SETTINGS_GLM
        VALUES (
                  DBMS_DATA_MINING.GLMS_REFERENCE_CLASS_NAME,
                  'GLMS_RIDGE_REG_DISABLE');

   COMMIT;
END;
```

The model was then created with `DBMS_DATA_MINING.CREATE_MODEL` (`CLASSIFICATION`, case id `ID`, target `가뭄`), and test accuracy was computed as the match rate between `PREDICTION(MD_CLASSIFICATION_MODEL4 USING *)` and the true label. The table name `SETTINGS_GLM` is a leftover from reusing a GLM template with only `ALGO_NAME` swapped. With `PREP_AUTO = ON`, Oracle handles preprocessing such as categorical encoding.

Train/test splits used `rownum`-based `id` thresholds. Scripts `02` and `03` cut the join output in its natural order at `id < 73000` and `id < 170000`; from `04` on, the table was shuffled with `ORDER BY DBMS_RANDOM.VALUE`, renumbered, and split at `id < 37700` (04) and `id < 25791` (05).

SQL gave accuracy cheaply, but per-class F1, AUC, or a tree-count curve each required a new query, and every hyperparameter change meant editing the settings table and repeating `DROP_MODEL`/`CREATE_MODEL`. That overhead drove the move to R.

### 3.4 Feature selection: 12 to 5

The twelve candidates were cumulative precipitation, storage, sunshine duration, outflow, mean ground temperature, mean temperature, dam name, mean relative humidity, total sunshine hours, season, province, and river basin. The selection rule in the slides (`docs/SPI6_가뭄예측_발표자료.pdf`) is "variables with high information gain that can be known at prediction time."

| Feature | Decision | Reason |
|---|---|---|
| cumulative precipitation, storage, sunshine duration, mean temperature | kept | high information gain, user-suppliable |
| effective storage | added | dam specification fixed once a dam is selected |
| outflow | dropped | ranked 4th, but it is an outcome of dam operation |
| mean ground temperature, dam name, mean relative humidity, total sunshine hours | dropped | no individual reason recorded |
| season, province, river basin | dropped | bottom three in the ranking |
| groundwater level | dropped | missing values shrink the sample |

The ranking order follows the slide figure; the information gain computation itself is not in the repo.

### 3.5 Scaling and split

Sunshine duration and mean temperature were min-max normalized; storage and cumulative precipitation were z-score standardized. The split is a 9:1 stratified split with `caret::createDataPartition(spi$가뭄, p = 0.9)` (`code/R/08_랜포_표준화_최종_F1_0.93.R`).

That script applies `normalize` and `standard` to the full `spi` frame before splitting. The dashboard code `code/R/preprocessing.R` instead stores the training-set min/max/mean/sd after the split and applies the same statistics to train, test, and user input.

### 3.6 R Shiny dashboard

The dashboard went through three versions: click-coordinate capture on a map PNG (`09_RShiny_UI_v1_지도클릭.R`), the same PNG rendered with plotly plus dam icons (`10_RShiny_UI_v2_Plotly.R`, `11_RShiny_UI_최종.R`), and a leaflet map (`preprocessing.R`, `etc/UI_v1.5.txt`). The final version places markers from dam latitude/longitude (`dam_loc.csv`) instead of image coordinates, pans to the selected basin or dam, and shows effective storage, total capacity, and height in `shinydashboard` valueBoxes. Selecting a dam fixes the effective storage input to that dam's value and caps the storage slider at its total capacity. On "predict", inputs are scaled with training statistics and the model returns class probabilities.

`code/R/preprocessing.R`

```r
    norm_precip_time <- normalize(input$precip_time, train_min_가조시간, train_max_가조시간)
    norm_avg_standard <- normalize(input$avg_standard, train_min_평균기온, train_max_평균기온)
    std_storage <- standard(input$storage, train_mean_저수량, train_sd_저수량)
    std_accum_precip <- standard(input$accum_precip, train_mean_누적강수량, train_sd_누적강수량)
    std_eff_storage <- standard(eff_storage_value, train_mean_유효저수량, train_sd_유효저수량)
    
    # Create a new data frame for prediction
    new_data <- data.frame(
      가조시간 = norm_precip_time,
      평균기온 = norm_avg_standard,
      저수량 = std_storage,
      누적강수량 = std_accum_precip,
      유효저수량 = std_eff_storage
    )
    
    # Make prediction
    prediction <- predict(rf_model, new_data, type = 'prob')
```

Probabilities are drawn as a plotly bar chart, the argmax class is printed as text, and an SPI6 stage reference table sits in the lower-right corner of the map.

## 4. Experiments

### 4.1 Setup

- Validation: single holdout. Oracle used `id`-threshold splits; R used a stratified `createDataPartition(p = 0.9)` split. No cross-validation.
- Models: Oracle `ALGO_RANDOM_FOREST` (default settings, `PREP_AUTO = ON`) and R `randomForest` (default `mtry`).
- Hyperparameters:

| Script | seed | ntree | Note |
|---|---|---|---|
| `07_랜포_트리수별_F1_비교.R` | 13 | sweep 1..100 | accuracy and macro F1 curves |
| `08_랜포_표준화_최종_F1_0.93.R` | split 10, fit 14 | 46, 59 | comment: "seed=14, ntree=59 gives high drought F1" |
| `preprocessing.R` (dashboard) | 24 | 97 | train-statistics scaling |

### 4.2 Results

The table below is from the slides; the README reports the same F1, AUC, and accuracy. Run outputs of the training scripts are not in the repo.

| Metric | Before | After |
|---|---|---|
| Accuracy | 95.7% | 96.1% |
| Precision | 0.941 | 0.944 |
| Sensitivity | 0.910 | 0.922 |
| Specificity | 0.985 | 0.987 |
| F1 | 0.925 | 0.933 |
| AUC | 0.948 | 0.955 |
| Rows (train/test) | 32,239 (29,019/3,220) | 47,098 (42,392/4,706) |

The row counts match `SPI_랜포_테이블_최종.csv` and `SPI_예측테이블_최종.csv`, respectively.

### 4.3 Ablation

Oracle-stage accuracies are recorded in the result CSV names listed in the header comment of `04_랜포_저수량포함_최종.sql`.

| Result file | Period | Mean storage | Accuracy |
|---|---|---|---|
| `spi18-22랜포(90퍼)저수량평균X.csv` | 2018-2022 | no | 90% |
| `spi19-22랜포(89퍼)저수량평균X.csv` | 2019-2022 | no | 89% |
| `spi19-22랜포(88퍼)저수량평균O.csv` | 2019-2022 | yes | 88% |
| `spi랜포14-22(85퍼)저수량평균O.csv` | 2014-2022 | yes | 85% |

The script's input table holds stage, cumulative precipitation, season, sunshine duration, mean temperature, province, and storage. The result CSVs themselves are not in the repo, so the exact configuration of each row cannot be confirmed beyond the file name.

- Extending the training period (2019-2022 to 2014-2022) did not improve accuracy.
- Adding storage did not improve accuracy in Oracle (89% to 88%). The final R model kept storage anyway.

R-stage configurations survive as datasets:

| Dataset | Inputs | Rows | Result |
|---|---|---|---|
| `SPI_랜포_테이블_최종.csv` | 12 candidates | 32,239 | before, F1 0.925 |
| `댐제외_데이터.csv` | cumulative precip, sunshine, temperature, groundwater | 81,239 | no score recorded |
| `수위제외_데이터.csv` | cumulative precip, sunshine, temperature, storage, dam | 47,131 | input to `08`; F1 0.93 per file name |
| 5 features + scaling | cumulative precip, storage, sunshine, temperature, effective storage | 47,098 | after, F1 0.933 |

Feature set, scaling, and sample size (32,239 to 47,098) all changed between "before" and "after". The repo contains no single-change experiment that would attribute the F1 gain from 0.925 to 0.933 to any one factor.

## 5. Limitations and future work

1. Scaling leakage: `08` scaled with full-data statistics before splitting. The effect on a random forest is likely small, but the procedure is wrong; the evaluation script should reuse the train-only transform already present in the dashboard code.
2. Selection on the test set: tree count (1 to 100) and seed (seed 14, ntree 59) were chosen on the same test-set F1, so the reported score may be optimistically biased. Selection should use a validation split or k-fold averages.
3. Train/serve mismatch: the dashboard model (seed 24, ntree 97) is not guaranteed to be the presented model. The dashboard also standardizes user-entered effective storage while the training data keeps it in raw units.
4. Temporal structure ignored: daily rows were split at random, so adjacent days land in both train and test. Neighboring days share nearly identical SPI6 and 6-month cumulative precipitation; a year-based temporal split or a station-grouped split is needed.
5. Feature-target overlap: SPI6 and the cumulative precipitation input both derive from roughly six months of rainfall. Predicting a future stage from lagged features only would be closer to a real forecasting task.

## References

- Code: https://github.com/tjehdgus/K_water_randomforest
- Development notes (Tistory, Korean): https://mynews76154.tistory.com/20
- Models and tools: Random Forest (Breiman, 2001), Oracle Data Mining `DBMS_DATA_MINING`, R `randomForest`, `caret`, `pROC`, R Shiny, `leaflet`
- Index: Standardized Precipitation Index (McKee et al., 1993)
