---
title: "Validation lessons from eight ML competitions: what to settle before choosing a model"
description: "Car classification, power and wind forecasting, sleep prediction, RAG and a clinical agent. The competitions I entered, summarised on the same template, and the validation rules I kept relearning."
date: 2026-10-15
lang: en
key: competition-lessons
tags: [ML competitions, Validation, Data leakage, Calibration, Time series]
tistory: https://mynews76154.tistory.com/27
---

Before every competition I read past solutions and retrospectives for similar tasks. What helps most is rarely which model won; it is which validation scheme people used and what did not work. So I summarised my own competitions on the same template and collected the lessons that kept coming back.

## Entries

| Competition | Task · Metric | Final approach | Result |
|---|---|---|---|
| Dacon HAI! Hecto AI Challenge (2025) | 396-class car images / LogLoss | ConvNeXt V2 + EMA·SWA, temperature scaling | 36th of 748 teams |
| Dacon power demand forecasting (2025) | Hourly demand, 100 buildings / SMAPE | Best of XGBoost·LightGBM·CatBoost per building, weighted average | 18th of 934 teams |
| ETRI Human Understanding AI Paper Challenge (2026) | Seven sleep metrics / mean log-loss | Per-target blend of nine sources, forward-chaining re-weighting | Private 22nd, paper accepted at ICTC 2026 |
| K-water Water Big Data Contest (2024) | Drought stage classification / F1 | Random forest on 5 variables + R Shiny | Honourable mention |
| Sejong City Big Data Contest (2024) | E-bike dock siting | Per-district K-Means, then MCLP or OPTICS | Finalist |
| Dacon BARAM 2026 wind power | Hourly output / 1-NMAE and FICR | MOS-corrected wind speed + BiGRU seq2seq blend | Early public 0.64526 |
| Fair Trade AI & Data Competition (2026) | Decision RAG / R@5·MRR·BERTScore·F1 | Hybrid retrieval + two rerankers + EXAONE LoRA | Internal eval 67.76 |
| AI Drug Discovery Challenge (2026) | Clinical trial failure early-warning agent | LangGraph multi-agent + XGBoost + RAG | Stop-probability AUC 0.730 |

## With small data, validation design decides the score

The ETRI sleep challenge had 450 training nights from 10 subjects. Optimising blend weights continuously improved OOF to 0.5992 but worsened the leaderboard to 0.6231. A 0.067 gain from feature selection vanished entirely when re-measured with nested CV, because selection and evaluation had shared folds. Per-entity models that worked in the power competition (about 2,000 rows per building) did worse than the prior with roughly 40 rows per subject.

With small data, keep weights coarse and do every selection step outside the evaluation folds.

## In time-series competitions, first find where the test set sits

In the power competition demand climbed all summer, and tree models, which cannot extrapolate beyond the training range, underestimated late August. In ETRI the evaluation nights were concentrated after the training nights; re-choosing weights with forward-chaining CV that mimicked this gave the biggest single improvement. For wind power I built year-based validation that mimicked the test structure and a harness that scored with the exact competition formula from day one, and local and leaderboard scores stayed correlated at about 0.91.

## If the metric scores probabilities, calibration is part of the model

Car classification used LogLoss, so after choosing a backbone I spent time on temperature scaling, EMA and SWA. But I set the temperature and ensemble weights on a single 20% hold-out, and private ended up 0.008 above public. Next time I would fit them on K-fold OOF predictions. In the clinical agent, probabilities were shown to users, so I measured ECE (0.018) alongside AUC.

## If you build the eval set yourself, draw the leakage line first

In the Fair Trade RAG competition every decision was made on a 278-question eval set we built ourselves. It leaked into embedding and reranker training, produced a 72.96, and we voided it ourselves ([write-up](/en/posts/kftc-rag/)). In K-water I picked seeds and tree counts on the same test-set F1, which in hindsight is the same mistake. Now I split dev and final eval sets from the start and keep both eval questions and their answers out of training.

## In short

1. Build a harness that scores with the exact metric before building models.
2. Find where the test set sits in time and shape CV to match.
3. With small data, move every selection step outside the evaluation folds.
4. If the metric uses probabilities, fit calibration on OOF predictions.
5. Never rely on a single eval set.
