---
title: "Pre-Quote Loss Check Simulator: Reframing Cost-Ratio Prediction as Unit-Cost Regression with RandomForest and a Recent-History Geometric Blend"
description: "A cost-ratio simulator that lets sales reps at a manufacturer check whether a deal will lose money before quoting. The model predicts unit cost, which does not depend on price, and the cost ratio is computed by dividing it by the entered price."
date: 2026-10-16
lang: en
key: cost-simulator
tags: [RandomForest, Regression, Feature Engineering, Simulation, FastAPI, Azure]
tistory: https://mynews76154.tistory.com/6
---

## Abstract

At my previous company, a manufacturer, I built a cost-ratio simulator for sales reps between January and February 2026. It takes an item, customer, department, sales rep, unit price, and quantity, and returns the predicted cost ratio, a loss flag, a risk grade, and the price needed to break even. Predicting the cost ratio directly gives no guarantee that the model responds consistently as the user changes the price, so the target was redefined as unit cost (cost of goods sold / quantity), which is independent of price, and the cost ratio was computed as `predicted unit cost / entered price`. The model is a RandomForest regressor whose output is blended with a weighted moving average of the three most recent transactions for the same item-customer pair via a geometric mean. Customers with no history are routed to a separate model that uses only four item-level features. The service ran on FastAPI in Azure Container Apps, and a February revision added user management and centralized configuration. The repository contains no training or validation logs, so no quantitative scores are reported here.

## 1. Problem

Sales reps needed to answer one question before sending a quote: "If I sell this item to this customer at this price and quantity, is it a loss?" By company rule, a transaction with a cost ratio of 85% or higher is classified as a loss.

| Item | Description |
|---|---|
| Input | Item code, customer, department, sales rep, unit price, quantity (new customers: item code, unit price, quantity) |
| Model output | Predicted unit cost |
| Derived outputs | Predicted cost ratio, loss flag, 5-level risk grade, cost ratio at +10/20/30% price, break-even price and required increase, quantity × price grid |
| Decision rule | `loss = cost_ratio >= 85` |

```text
cost_ratio(%)       = cost_of_goods_sold / revenue * 100
unit_cost           = cost_of_goods_sold / quantity
pred_cost_ratio(%)  = pred_unit_cost / entered_price * 100
break_even_price    = pred_unit_cost / 0.85
```

There were two constraints. Users sweep the price repeatedly, and a model trained on cost ratio is not guaranteed to keep the inverse relationship between price and cost ratio in price ranges absent from the training data. The output also had to be explainable to sales reps. With unit cost as the target, the explanation reduces to "the model estimates this deal's unit cost at X, which is Y% of the price you entered."

## 2. Data

The source was an internal sales-history table exported as CSV. Raw data and trained model files are excluded from the repository (`data/` in `.gitignore`), so row counts and time span cannot be confirmed and are not reported. The schema and preprocessing below come from the loading code in `main.py`.

| Group | Columns |
|---|---|
| Identifiers / categorical | Item code, item name, item category (3 levels), customer, department, sales rep, sales unit |
| Numeric | Unit price, quantity, sales amount (KRW), cost of goods sold, cost ratio (%) |
| Time | One period column (sort key) |

Preprocessing steps:

1. Drop rows with an empty customer field
2. Coerce price and quantity to numeric and drop failures
3. Keep rows where price, quantity, and sales amount are all positive
4. Sort by period, then item code
5. Derive an `item_customer` key, `loss = cost_ratio >= 85`, and `unit_cost = cost_of_goods_sold / quantity`

At startup the server builds group statistics, lag tables, moving-average tables, and a relation map for dropdowns (item → customer → department → sales rep), all held in memory. The relation map narrows each dropdown so that users cannot enter a combination that never occurred.

## 3. Method

### 3.1 Pipeline

```text
[Server startup]
Sales history CSV
  → preprocessing / derive unit_cost and loss flag
  ├─ group stats (item, customer, item×customer, department, sales rep)
  ├─ time-series stats (lag 1-3, MA 3/5/10, moving std 3/5)
  └─ relation map (item → customer → department → sales rep)

[Prediction request]
Input (item, customer, department, sales rep, price, quantity)
  → feature assembly (stat-table lookup + label encoding + log quantity + calendar)
  → RandomForest unit-cost prediction
  → geometric blend with recent 3-transaction WMA (model value only if no history)
  → cost_ratio = pred_unit_cost / price * 100
  ├─ loss flag, risk grade, break-even price
  ├─ price-increase simulation (no model call)
  ├─ quantity × price grid (one model call per quantity row)
  └─ LLM commentary (gpt-4o-mini)
```

- Stat tables: precomputed group summaries of past transactions so features can be assembled at request time.
- Feature assembly: missing lookups fall back to substitute values (3.3).
- Unit-cost prediction: routed to either the existing-customer or the new-customer model.
- Cost-ratio computation: price is used only here, never as a model input.

### 3.2 Reframing the target: unit cost instead of cost ratio

The variable the user manipulates, price, was moved out of the model. The model predicts only unit cost, and the cost ratio and everything derived from it are computed by division.

`main.py`
```python
        판매금액 = 단가 * 수량
        예측_원가율 = (예측_단위원가 / 단가) * 100
        적자여부 = 예측_원가율 >= PROFIT_THRESHOLD
        
        if 예측_원가율 >= 100: 위험등급 = "매우위험"
        elif 예측_원가율 >= 90: 위험등급 = "고위험"
        elif 예측_원가율 >= 85: 위험등급 = "주의"
        elif 예측_원가율 >= 75: 위험등급 = "관심"
        else: 위험등급 = "안전"
        
        원가율_10up = (예측_단위원가 / (단가 * 1.1)) * 100
        원가율_20up = (예측_단위원가 / (단가 * 1.2)) * 100
        원가율_30up = (예측_단위원가 / (단가 * 1.3)) * 100
        필요단가 = 예측_단위원가 / 0.85
        필요인상률 = ((필요단가 / 단가) - 1) * 100
```

(Identifiers are in Korean: `단가` = price, `예측_단위원가` = predicted unit cost, `예측_원가율` = predicted cost ratio, `필요단가` = break-even price.)

Under this design, the cost ratio is strictly decreasing in price, and price simulations never call the model again. Quantity does affect unit cost through `log_수량` (log quantity), so the quantity × price grid calls the model once for each of five quantity rows (0.5, 1, 2, 5, 10×) and fills the six price columns (-20% to +30%) by division.

### 3.3 Features

The inference code for the existing-customer model assembles 40 candidate features in five groups, then selects them in the order stored in the artifact's `feature_cols`.

| Group | Features | Count |
|---|---|---|
| Categorical encoding | Label-encoded item, customer, department, sales rep (unseen → -1) | 4 |
| Quantity / calendar | `log1p(quantity)`, year, month, quarter, month sin/cos | 6 |
| Group stats | Mean unit cost, std, transaction count per item, customer, sales rep, department, item×customer | 15 |
| Lags | Item lag1-3, customer lag1-2, item×customer lag1 | 6 |
| Rolling stats / trend | Item MA3/5/10, item MSTD3/5, customer MA3/5, item and customer trend (lag1 - mean) | 9 |

Item-level mean unit cost, std, and mean cost ratio exclude the last transaction. Using a mean that includes the last transaction to predict that same transaction would leak the target into the features.

`main.py`
```python
def calc_품번_stats(group):
    if len(group) > 1:
        평균단위원가 = group['단위원가'].iloc[:-1].mean()
        단위원가_std = group['단위원가'].iloc[:-1].std()
    else:
        평균단위원가 = group['단위원가'].iloc[0]
        단위원가_std = 0
    return pd.Series({
        '품번_평균단위원가': 평균단위원가,
        '품번_단위원가_std': 단위원가_std,
        '품번_거래건수': len(group),
        '품번_최근단위원가': group['단위원가'].iloc[-1],
        '품번_평균원가율': group['원가율(%)'].iloc[:-1].mean() if len(group) > 1 else group['원가율(%)'].iloc[0],
        '품번_최근원가율': group['원가율(%)'].iloc[-1],
    })
```

Missing values are filled hierarchically. Missing item lags and moving averages fall back to the item mean, missing customer lags to the customer mean, missing groups to the global mean unit cost, and any remaining NaN becomes -999.

### 3.4 Geometric blend with recent history

Because RandomForest relies heavily on group-mean features, it gets pulled toward the historical mean for pairs whose cost moved recently. To correct for this, a weighted moving average (weights 1, 2, 3, newest weighted 3) of the last three unit costs for the item-customer pair is combined with the model prediction via a geometric mean.

`main.py`
```python
    품번_거래처_recent = df[(df['품번'] == 품번) & (df['거래처'] == 거래처)].sort_values('년월', ascending=False)
    
    if len(품번_거래처_recent) >= 1:
        recent_values = 품번_거래처_recent['단위원가'].head(3).values
        if len(recent_values) == 1:
            품번_거래처_wma3 = recent_values[0]
        elif len(recent_values) == 2:
            품번_거래처_wma3 = np.average(recent_values, weights=[2, 3][-len(recent_values):])
        else:
            품번_거래처_wma3 = np.average(recent_values, weights=[1, 2, 3])
        예측_단위원가 = (품번_거래처_wma3 ** 0.5) * (예측_단위원가_rf ** 0.5)
        예측방식 = "Hybrid 기하평균 (최근값^0.5 × 모델^0.5)"
    else:
        예측_단위원가 = 예측_단위원가_rf
        예측방식 = "모델 Only"
```

A geometric mean was used instead of an arithmetic mean because unit costs differ by orders of magnitude across items, and blending on a ratio scale is less sensitive when one of the two values spikes. The response carries a `예측방식` (prediction method) field so the UI shows whether the result is blended or model-only.

### 3.5 New-customer model

Feeding a customer with no history into the existing model fills 21 customer-, department-, and rep-related features with substitutes such as the unseen-label code (-1) or the global mean. The splits the model learned from differences between customers would then receive meaningless inputs, so a separate model using only item information was trained and shipped as a second artifact.

`main.py`
```python
    features = pd.DataFrame([{
        '품번_encoded': 품번_encoded,
        'log_수량': np.log1p(수량),
        '품번_평균단위원가': 품번_평균단위원가 if not pd.isna(품번_평균단위원가) else 전체_평균단위원가_new,
        '품번_거래건수': 품번_거래건수 if not pd.isna(품번_거래건수) else 0,
    }])

    features = features[feature_cols_new]
    예측_단위원가_rf = model_new.predict(features)[0]
```

The blend works the same way, but uses the last three transactions for the item across all customers instead of the item-customer pair. The "new customer" tab calls dedicated endpoints, and its detailed analysis shows item history only.

### 3.6 LLM commentary

Each prediction is accompanied by commentary from gpt-4o-mini on Azure OpenAI (`max_tokens=1500`, `temperature=0.5`). The prompt contains the predicted cost ratio, the price-increase simulation, historical loss frequency for the item and customer, and the item's per-customer and the customer's per-item summaries (top 15 each). The model is asked to write four sections: overall assessment, item analysis, customer analysis, and simulation notes. The system prompt requires keeping units (% vs. KRW) distinct, and the deal's predicted cost ratio is repeated at the start and end of the prompt. Prescriptive phrasing ("you should raise the price") is forbidden so the decision stays with the sales rep, and loss rates must be phrased as "M of N transactions (X%)" so they are not confused with a cost ratio or probability.

### 3.7 Serving and user management

The backend is FastAPI and the frontend is a single HTML page. The Docker image was deployed to Azure Container Apps (0.5 vCPU, 1 GiB, 1-3 replicas), and model files and sales data are read from an Azure File volume mounted at `DATA_DIR`, so refreshing data does not require rebuilding the image.

The February revision added a shared user-management module used across other internal tools, and consolidated environment variables into `config.py`. Users, permissions, and sessions are stored in Cosmos DB.

| Item | Implementation |
|---|---|
| Passwords | bcrypt; legacy SHA-256 hashes are re-hashed to bcrypt on successful login |
| Password policy | 8+ characters with at least one uppercase, lowercase, digit, and special character |
| Login throttling | 15-minute lockout after 5 failures |
| Authorization | Per-app permission records, role levels `admin > staff > viewer` |
| Audit log | Login success, failure, lockout, and hash migration events |
| Admin password | Startup fails in production if unset; a temporary password is generated locally |

## 4. Experiments

### 4.1 Setup

The repository contains only the inference server; training scripts, validation splits, hyperparameters, and evaluation logs are not included. Trained models were shipped as two pickle artifacts (existing and new customers), each holding `model`, `label_encoders`, and `feature_cols`. Both are RandomForest regressors targeting unit cost. The existing-customer model uses `feature_cols` out of 40 candidates; the new-customer model uses four features. The blend reference is the last three item-customer transactions and the last three item transactions respectively, with the geometric-mean exponent fixed at 0.5 in both. This section documents per-endpoint behavior that can be verified from the code.

### 4.2 Results

Quantitative metrics (R², MAPE, etc.) are not reported because the repository provides no evidence for them. Per-endpoint computation verified from the code:

| Function | Endpoint | Model calls | Other computation |
|---|---|---|---|
| Single prediction | `/api/predict_cost_rate` | 1 | Cost ratio, risk grade, +10/20/30%, break-even price |
| Custom price simulation | `/api/simulate_custom_price` | 0 | One division |
| Quantity × price grid | `/api/simulate_price_quantity_grid` | 6 (5 quantity rows + 1 baseline) | Cost ratio and verdict for 30 cells |

### 4.3 Ablation

No quantitative comparison of the variants below remains in the repository. The table summarizes, from the code, how each design choice changes simulator behavior.

| Variant | Model re-call on price change | Cost ratio monotone in price | Note |
|---|---|---|---|
| Direct cost-ratio regression | Required | Not guaranteed | Not adopted |
| Loss/profit classification at 85% | Required | Not guaranteed | Cannot produce a break-even price |
| Unit-cost regression + division (adopted) | Not required | Guaranteed | Break-even price in closed form |
| Model only | - | - | Recent cost shifts pulled toward group means |
| Model + recent 3-transaction WMA, geometric mean (adopted) | - | - | Applied only when pair history exists |
| New customers through the existing model | - | - | 21 customer-side features become substitutes |

## 5. Limitations and future work

- **Reproducibility.** Without training and validation code or logs in the repository, performance and variant comparisons cannot be reproduced.
- **Timing of stat tables.** Item means exclude the last transaction, but customer stats, lags, and moving averages include the most recent one. That is correct for predicting new deals in production, but re-predicting past transactions with the same tables for evaluation would leak targets. Evaluation needs tables rebuilt from data before each cutoff.
- **Year extrapolation.** Year and month features are taken from the request time. Beyond the last year in the training data, a tree model effectively treats the input as that last year. Dropping the year or replacing it with elapsed time would be better.
- **Fixed blend exponent.** The geometric-mean exponent of 0.5 is hard-coded, and the repository has no validation record for choosing it. It should vary with item-level cost volatility, or at least candidate values should be compared with time-based validation.
- **Serving.** Some simulation, detailed-analysis, and LLM endpoints lack login checks, and failed-login counts are kept in process memory, so lockouts are not shared across replicas. Authentication should be unified as a dependency and attempt records moved to shared storage.

## References

- Tistory development log: https://mynews76154.tistory.com/6
- Random Forests (Breiman, 2001)
- FastAPI, Azure Container Apps, Azure Cosmos DB
- Azure OpenAI gpt-4o-mini
