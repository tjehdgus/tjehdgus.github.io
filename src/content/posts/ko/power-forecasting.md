---
title: "건물 100개 시간별 전력 예측: 도메인 피처와 건물별 GBDT 모델 선택 앙상블"
description: "데이콘 2025 전력사용량 예측 AI 경진대회에서 CDH·THI·건물별 휴무일 피처와 패턴 매칭 스플라인 보간을 만들고, 건물마다 XGBoost·LightGBM·CatBoost 중 검증 SMAPE가 가장 낮은 모델을 골라 Public LB SMAPE 5.22를 얻었다."
date: 2026-03-17
lang: ko
key: power-forecasting
tags: [Time Series, Energy Forecasting, Feature Engineering, XGBoost, LightGBM, CatBoost, Dacon]
tistory: https://mynews76154.tistory.com/18
---

## 초록

데이콘 「2025 전력사용량 예측 AI 경진대회」(2025.08, 2인 팀)에서 건물 100개의 2024-08-25~08-31 시간별 전력소비량을 예측했다. 평가 지표는 SMAPE다. 전처리로 0값과 지정 이상치를 같은 시간대·휴일 조건의 다른 주 값으로 CubicSpline 보간했고, 냉방도시(CDH), 온습도지수(THI), 백화점 16개 건물의 개별 휴무 규칙, 건물별 요일-시간대 전력 통계를 피처로 만들었다. 모델은 건물마다 따로 학습한 XGBoost, LightGBM, CatBoost이며, 7일 단위 역순 12-Fold로 검증했다. 건물별로 세 모델 중 검증 SMAPE가 가장 낮은 모델을 고르는 방식(Comparison)이 Public LB 5.22로 가장 좋았고, 단일 CatBoost는 5.29539, Ridge Stacking은 5.2633이었다. 최종 제출은 리더보드 기준 후보의 가중평균이며 최종 순위는 934팀 중 18위(보고된 값)다.

## 1. 문제 정의

- 과제: 건물 100개의 시간별 전력소비량(kWh) 예측
- 입력: 건물별 시간 단위 기상(기온, 강수량, 풍속, 습도), 건물 정보(유형, 연면적, 냉방면적, 태양광·ESS·PCS 용량)
- 출력: 2024-08-25 00시 ~ 08-31 23시, 건물당 168시간, 총 16,800개 예측값
- 평가 지표: SMAPE (낮을수록 좋음)

```text
SMAPE = (100 / n) * Σ |F_t - A_t| / ((|A_t| + |F_t|) / 2)
```

SMAPE는 상대 오차라서 소비량이 작은 건물도 큰 건물과 같은 무게로 반영되고, 휴무일을 평일로 예측하는 식의 큰 비율 오차 몇 시간이 점수를 크게 깎는다.

제약 조건으로 테스트 기간에는 일조·일사 값이 없어 두 변수는 학습에도 쓰지 않았고, 학습 기간은 여름 한 시즌(6~8월)뿐이며 테스트는 그 바로 뒤 7일이다.

## 2. 데이터

| 파일 | 내용 |
|---|---|
| `train.csv` | 2024-06-01 ~ 08-24, 100개 건물의 시간별 기상과 전력소비량 |
| `building_info.csv` | 건물 유형, 연면적, 냉방면적, 태양광·ESS·PCS 용량 |
| `test.csv` | 2024-08-25 ~ 08-31 기상 (일조·일사 없음) |
| `sample_submission.csv` | `건물번호_일시` ID와 예측값 |

학습 기간은 85일이라 건물당 2,040행, 전체 204,000행이다. 건물 유형은 10종(백화점 16개, 나머지 유형 8~10개)이고, 발표 자료의 EDA 표 기준 유형별 평균 소비량은 IDC 10316.94kWh에서 아파트 1106.31kWh까지 약 9배 차이가 난다.

EDA 결과 건물마다 소비 곡선이 크게 달라 건물별 모델링을 기본으로 했다. 패턴 안의 불규칙한 변동은 테스트 기간에도 나올 수 있다고 보고 남겼고, 추세에서 명백히 벗어난 점만 이상치로 보간했다.

### 2.1 전처리

`building_info`를 건물번호로 조인하고, 전력소비량에 두 종류의 보정을 했다.

| 대상 | 정의 | 보간 기준 |
|---|---|---|
| 지정 이상치 | `SPECIFIC_OUTLIERS`에 건물·날짜·시간을 직접 기록. 25개 건물, 62개 시점 | 휴일 시점은 같은 시간·같은 요일의 휴일 값, 평일 시점은 같은 시간의 휴일 값 |
| 0값 | `power_consumption < 1e-10` | 같은 시간, 같은 요일, 같은 휴일 여부 |

조건이 맞는 다른 주의 값을 모아 "학습 시작 후 경과 주차"를 x축으로 natural CubicSpline을 맞추고, 대상 시점의 주차에서 값을 읽는다. 여름 동안 소비량이 우상향하므로 단순 중앙값보다 주차 추세를 따라가는 값이 나온다. 조건을 만족하는 점이 4개 미만이면 같은 시간대 중앙값, 같은 요일 중앙값, 건물 전체 중앙값 순으로 대체한다.

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

### 2.2 피처

`create_features()`가 만드는 피처는 다음과 같다.

| 범주 | 피처 | 정의 |
|---|---|---|
| 시간 순환 | `sin/cos_hour`, `sin/cos_date`, `sin/cos_month`, `sin/cos_dayofweek` | 시(÷23), 월+일/31(÷12), 월(÷12), 요일(÷7) |
| 계절 | `summer_sin/cos` | 2024-06-01 ~ 09-14를 한 주기로 본 위상 |
| 휴무 | `holiday`, `week_of_month` | 건물별 휴무 규칙, 격주 일요일 플래그 |
| 기온 통계 | `day_max/min/mean_temperature`, `day_temperature_range`, `enhanced_avg_temp` | 건물·일 단위 집계, 3시간 간격 값의 일평균 |
| 복합 지수 | `CDH`, `THI`, `WCT` | 냉방도시, 온습도지수, 체감온도 |
| 기상 플래그 | `weather`, `hot_humid_condition`, `hours_since_rain` | 강수 전후 ±3시간, 26도 초과·습도 70% 초과, 마지막 강수 후 경과 시간(최대 168) |
| 전력 통계 | `day_hour_mean/std`, `holiday_hour_mean/std`, `hour_mean/std` | 건물별 (시간, 요일) / (시간, 휴일) / 시간 단위 학습 기간 평균·표준편차 |
| 건물 | `total_area`, `cooling_area`, `cooling_ratio`, `building_size_category` | 면적, 냉방면적 비율, 1천·1만 m2 기준 3단계 |
| 시계열 | `temp_rolling_3h_mean`, `temp_rolling_6h_max`, `temp_lag_1h/3h`, `temp_diff_1h`, `humidity_lag_1h` 등 | 건물별 롤링·lag |
| 비선형 | `temperature_squared`, `humidity_squared`, 기온×시간, 기온×습도 | 제곱항과 상호작용 |

## 3. 방법

### 3.1 전체 파이프라인

```text
train.csv / test.csv / building_info.csv
  │
  ├─ [1] 로딩·조인: 컬럼 영문화, 건물 정보 merge, 태양광/ESS 플래그
  ├─ [2] 휴무 규칙: 건물별 holiday 계산 (백화점 16개 개별 규칙 + 일반 규칙)
  ├─ [3] 보간: 지정 이상치 62점 → 0값, 패턴 매칭 CubicSpline (train만)
  ├─ [4] 피처: 시간 순환, 계절, 기온 통계, CDH/THI/WCT, 전력 통계, 롤링/lag
  │
  ├─ [5] 건물별 학습 (100개 건물 × 3 모델)
  │     ├─ XGBoost   12-Fold × 5 seed = 60개 평균, 과소추정 가중 MSE
  │     ├─ LightGBM  12-Fold = 12개 평균, 2단계 가중치 재학습
  │     └─ CatBoost  12-Fold × 5 seed = 60개 평균, log 타깃, 2단계 가중치 재학습
  │
  ├─ [6] Comparison: 건물마다 검증 SMAPE 최저 모델의 예측 선택
  ├─ [7] Stacking (비교용): 세 모델 예측 → Ridge 메타모델
  └─ [8] 최종 제출: LB 기준 후보 선정 후 가중평균
```

[1]~[4]는 `code/final.py`에 남아 있다. [5]~[8]은 학습 코드가 레포에 없어 설정과 점수를 발표 자료(`docs/한국에너지공단.pdf`) 기준으로 적었다.

### 3.2 냉방 부하 지수: CDH, THI

CDH는 최근 12시간 동안 기온이 26도를 넘은 양의 누적합으로, 같은 시각 같은 기온이라도 더위가 몇 시간째 이어졌는지를 구분한다. 건물당 학습 데이터가 12주 남짓이라 트리 모델이 lag 조합으로 이 효과를 배우게 하기보다 누적 지표를 직접 넣었다. THI는 기온과 습도, WCT는 기온과 풍속으로 계산한다.

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

### 3.3 건물별 휴무 규칙

일반 건물은 주말과 공휴일(06-06, 06-07, 08-15, 08-16)을 휴무로 처리했다. 백화점은 건물마다 휴무일이 달라 건물별 그래프에서 소비량이 급락하는 날을 찾아 규칙으로 옮겼다. 코드에 개별 규칙이 있는 건물은 16개다. 매주 일요일(18번), 격주 일요일(27·40·59·63번), 매월 10일과 특정 일요일(29번), 격주 월요일(32번), 특정 월요일(19·45·54·79·95번), 격주 일요일과 추가 월요일(74번), 연중무휴(34·73·88번)로 나뉜다.

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

규칙에는 테스트 기간의 휴무일(08-25 일요일, 08-26 월요일 등)까지 포함했다. 같은 `holiday` 값이 보간의 패턴 매칭 조건과 `holiday_hour_mean/std` 통계의 그룹 키로도 쓰여, 휴무 규칙 하나가 전처리·피처·예측 전반에 반영된다.

### 3.4 건물별 기준선: 전력 통계 피처

건물별로 (시간, 요일), (시간, 휴일 여부), 시간 단위의 학습 기간 평균·표준편차를 학습·테스트 행에 붙였다. 건물의 평소 소비 수준을 기준선으로 주고, 기상 피처로 기준선에서 벗어나는 양을 학습하게 하려는 설계다.

### 3.5 모델과 과소추정 보정

세 모델 모두 건물마다 따로 학습했다. 발표 자료에 따르면 XGBoost가 소비량을 낮게 예측하는 경향이 뚜렷했고, 이를 모델별로 보정했다.

- XGBoost: 실제보다 낮게 예측한 샘플에 100배 가중치를 주는 Weighted MSE. 요일별 SMAPE를 따로 측정했다.
- LightGBM, CatBoost: 1단계 모델로 과소추정 패턴을 찾고, 2단계에서 샘플 가중치(alpha=1)를 넣어 재학습했다.
- CatBoost: 타깃을 `log(y + 1e-6)`로 변환해 학습했다. 발표 자료에는 추가 보정 없이 과대추정 경향을 보였다고 기록되어 있다.

### 3.6 건물별 모델 선택 (Comparison)

건물마다 세 모델을 모두 학습해 검증 SMAPE를 비교하고, 가장 낮은 모델의 테스트 예측을 그 건물의 예측으로 썼다. 발표 자료의 예시로 1번 건물은 CatBoost(10.10088), 2번은 LightGBM(4.995961), 98·99번은 XGBoost(9.265551, 3.136065)가 최저였다.

## 4. 실험

### 4.1 실험 설정

- 검증: 학습 기간 끝에서부터 7일 단위로 거꾸로 자른 12-Fold. 검증 구간 앞뒤 1일은 학습에서 제외했다.
- Seed: 2025, 42, 123, 777, 999 (XGBoost, CatBoost).
- 하드웨어: 레포에 기록이 없다.

| 모델 | 주요 하이퍼파라미터 | 앙상블 |
|---|---|---|
| XGBoost | lr 0.1, n_estimators 1000, max_depth 5, subsample 0.9, colsample_bytree 0.8, min_child_weight 6, reg_alpha 1, reg_lambda 1 | 12 fold × 5 seed = 60개 평균 |
| LightGBM | lr 0.05, n_estimators 2000, max_depth 5, num_leaves 50, subsample 0.9, colsample_bytree 0.8, min_child_samples 15, reg_alpha 1, reg_lambda 1 | 12 fold = 12개 평균 |
| CatBoost | lr 0.1, iterations 1000, depth 5, subsample 0.9, colsample_bylevel 0.8, log 타깃 | 12 fold × 5 seed = 60개 평균 |
| Stacking 메타 | Ridge (베이스: XGB 28개, LGBM 27개, CatBoost 32개 피처) | |

### 4.2 결과

Public LB SMAPE 기준이며 발표 자료의 앙상블 후보 표에서 가져왔다.

| 구성 | Public LB SMAPE |
|---|---|
| CatBoost 단일 (건물별) | 5.29539 |
| Ridge Stacking | 5.2633 |
| **Comparison v2 (건물별 최고 모델 선택)** | **5.22** |

최종 제출물은 리더보드 점수로 후보를 고른 뒤 가중평균한 앙상블이다. 가중치 구성과 최종 Private 점수는 레포에 기록되어 있지 않다. 최종 순위는 934팀 중 18위(보고된 값)다.

### 4.3 비교 실험

**건물별 모델 vs 유형별 모델**

| 모델 | 건물별 (100개 모델) | 유형별 (10개 모델) | 차이 |
|---|---|---|---|
| CatBoost | 5.29539 | 5.78100 | +0.486 |
| LightGBM | 5.39090 | 5.81680 | +0.426 |
| XGBoost | 5.39160 | 5.7983 | +0.407 |

세 모델 모두 유형별로 묶으면 0.4 이상 나빠졌다. 건물당 2,040행이면 트리 모델 학습에 충분했고, 같은 유형이라도 규모와 운영 방식이 달라 묶으면 서로 다른 곡선이 섞인다.

**앙상블 방식**

| 구성 | Public LB SMAPE | 비고 |
|---|---|---|
| CatBoost 단일 | 5.29539 | 기준 |
| CatBoost alpha값 앙상블 | 5.3765 | 기준보다 나쁨 |
| Comparison v1 | 5.56768 | 기준보다 나쁨 |
| Comparison v2 | 5.22 | 단일 모델 최고 파라미터, log 변환 |
| Ridge Stacking | 5.2633 | v2보다 0.043 나쁨 |

Comparison v1에서 v2로 바뀐 내용은 발표 자료에 "앞서 진행한 단일 모델의 최고 성능 파라미터 사용"과 "log(y + 1e-6) 변환"까지만 적혀 있다.

## 5. 한계와 향후 과제

1. **전력 통계 피처의 검증 누수.** `create_enhanced_power_statistics()`는 학습 기간 전체로 요일-시간대 평균·표준편차를 계산해 학습 행에 그대로 붙인다. 검증 폴드의 정답이 피처에 섞여 검증 SMAPE가 낙관적이었을 수 있고, 폴드마다 학습 구간만으로 계산해야 한다.
2. **건물별 모델 선택의 분산.** 건물 하나의 검증 구간은 폴드당 168시간이라, 세 모델 점수 차가 작은 건물에서는 선택이 잡음에 가깝다. 12개 폴드 평균으로 고르고 차이가 작으면 평균을 쓰며, 최종 가중치도 Public LB 대신 OOF로 정해야 한다.
3. **코드와 실제 학습 설정의 불일치.** `final.py`의 `feature_cols`는 `'humidity_squared'` 뒤 쉼표가 빠져 `'humidity_squaredtemp_rolling_3h_mean'`이라는 없는 컬럼이 생기고 항목이 44개가 된다. 발표 자료의 모델별 피처 수(28/27/32개)와도 달라, 학습에는 별도의 목록을 쓴 것으로 보인다. 학습 코드가 레포에 없어 [5]~[8] 단계는 재현할 수 없다.
4. **계산식 검증.** 코드의 THI는 `(9/5*humidity - 26)` 항에 기온 대신 습도를 쓰고, WCT는 풍속을 km/h로 바꾸지 않고 m/s 그대로 넣는다. 평일 이상치 보간도 패턴 조건이 `holiday == True`로 되어 있어 휴일 값으로 메워진다. 의도와 맞는지 확인하고 표준식으로 고친 뒤 기여도를 다시 측정해야 한다.
5. **트리 모델의 외삽.** 소비량이 우상향하는데 GBDT는 학습 범위 밖으로 값을 늘리지 못한다. 추세를 분리해 선형으로 예측하고 잔차만 GBDT로 학습하는 구조, Transformer 계열 모델과의 비교가 남아 있다.

## 참고

- 코드: https://github.com/tjehdgus/dacon_elec_catboost
- 개발 기록 (티스토리): https://mynews76154.tistory.com/18
- 모델: XGBoost, LightGBM, CatBoost, Ridge Regression
- 기법: Cubic Spline Interpolation, Cooling Degree Hours, Temperature-Humidity Index, Wind Chill Temperature
