---
title: "Cost-ratio simulator: tech stack and components"
description: "The technical makeup of a pre-quote cost-ratio simulator built at a previous company. Company data and internal figures are not covered."
date: 2026-04-07
lang: en
key: cost-simulator
tags: [RandomForest, Regression, Feature Engineering, FastAPI, Azure, LLM]
tistory: https://mynews76154.tistory.com/6
---

## Overview

At a previous company I built a web simulator that lets sales staff check the cost ratio of a deal, and whether it would lose money, before sending a quote. This post only lists which technologies were used and where. Data, model performance, business thresholds and internal system details are out of scope.

## Tech stack

| Area | Technology |
|---|---|
| Model | scikit-learn RandomForest regression |
| Data processing | pandas, NumPy |
| Backend | Python, FastAPI |
| Frontend | Single HTML page (JavaScript) |
| LLM | Azure OpenAI (result commentary) |
| Auth and storage | bcrypt password hashing, Azure Cosmos DB |
| Deployment | Docker, Azure Container Apps, Azure File volume |

## Architecture

```text
Sales records
  → preprocessing / derived columns
  → group and time-series statistic tables (built once at startup)

Prediction request
  → feature assembly
  → RandomForest regression
  → correction from recent records
  → cost-ratio calculation / simulation
  → LLM commentary
```

## Components

- **Problem framing.** Users vary the price, so price was kept out of the model. The model regresses a price-independent quantity, and the cost ratio is computed by dividing by the entered price. Price simulations need no extra model calls.
- **Feature engineering.** Label-encoded categoricals, log transforms, cyclical time features (sin/cos), group statistics, lags, and rolling means and standard deviations. Missing values fall back hierarchically to coarser group statistics.
- **Leakage control.** Group-average features were built so the target transaction does not leak into its own statistics.
- **Blended correction.** The model prediction is combined with a weighted moving average of recent records using a geometric mean.
- **Cold start.** Inputs with no history go to a separate model with a reduced feature set.
- **LLM commentary.** The prediction and a summary of past history are fed into a prompt that writes an explanation; directive phrasing is disallowed so the decision stays with the user.
- **Serving.** Prediction and simulation APIs in FastAPI, shipped as a Docker image to Azure Container Apps. Model and data are mounted from a volume so they can be swapped without rebuilding the image.
- **User management.** bcrypt hashing, password policy, login attempt limits, role-based permissions and audit logging.

## References

- Build log (Korean): https://mynews76154.tistory.com/6
- Random Forests (Breiman, 2001)
- FastAPI, Azure Container Apps, Azure Cosmos DB, Azure OpenAI
