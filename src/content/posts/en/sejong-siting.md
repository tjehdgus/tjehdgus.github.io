---
title: "Siting public e-bike docks in Sejong City: a travel-speed stress index, per-district K-Means with MCLP or OPTICS, and Seoul's Ttareungyi as a proxy ground truth"
description: "Built a deceleration-based road stress index from travel speeds on 24 roads to justify public e-bikes, then selected dock locations per administrative district with K-Means followed by MCLP or OPTICS depending on coverage. With no ground truth for siting, the method was checked against Seoul's Ttareungyi stations using the share of cluster centres within 200 m of a station."
date: 2026-10-16
lang: en
key: sejong-siting
tags: [Facility Location, K-Means, MCLP, OPTICS, Geospatial Analysis, Public Data]
tistory: https://mynews76154.tistory.com/22
---

## Abstract

For the 5th Sejong City Big Data Analysis Idea Competition (autumn 2024), we proposed per-district locations for docks of a hypothetical public e-bike system. Stage one quantified the case for e-bikes: travel speeds on 24 roads (39,336 rows, 417 road-segment-direction units) were turned into a deceleration-based road stress index. Stage two split bus stops, crosswalks, per-stop route counts and parking lots into administrative-district polygons, ran K-Means per district, and then kept the MCLP solution for the 8 districts whose MCLP coverage exceeded 70% and an OPTICS solution for the remaining 6. Because siting has no ground-truth map, the variable set and method were validated in Seoul: K-Means centres computed from subway stations, bus stops and crosswalks were scored by the share lying within 200 m of an actual Ttareungyi (Seoul public bike) station. Across the 14 districts, K-Means silhouette scores ranged from 0.52 to 0.78 and MCLP coverage from 21% to 88%. The entry advanced to the final round.

## 1. Problem

The task was to decide where, in each administrative district (dong), Sejong should place docks for a public e-bike system. As a policy-proposal competition, the entry had to answer two questions.

1. Why e-bikes? Show, with numbers, how many road segments in a wide-roaded planned city actually slow down.
2. Where should the docks go? For each district, take candidate point coordinates as input and output dock locations (cluster centres or selected candidate points).

Each stage used its own metric.

```text
deceleration(segment, slot) = 1 - (mean speed in slot / segment top speed)
silhouette score            = K-Means cluster quality (accept if > 0.5)
coverage (%)                = points in clusters containing a selected site / all points * 100
Ttareungyi proximity (%)    = mean( min_dist(centre, Ttareungyi station) < 0.002 deg ) * 100
```

The central constraint was the lack of labels. Sejong has no e-bike docks yet, so there is no way to check whether a plausible-looking clustering puts docks in the right place. That is why a separate validation stage was built on Seoul's Ttareungyi network as a stand-in for ground truth.

## 2. Data

| Data | Source file | Size | Used for |
|---|---|---|---|
| Segment travel speeds | `data/road_speed/*.xls` | 24 road files | stress index |
| Preprocessed speeds | `data/processed/세종시_구간통행속도_전처리_최종.csv` | 39,336 rows; 417 road-segment-direction units; 5 weekdays; 19 hourly slots | stress index |
| Sejong crosswalks | `data/raw/세종시_횡단보도.csv`, `조치원_횡단보도.csv` | 4,970 and 680 rows | siting variable |
| Bus stops by route | `data/raw/세종시_노선별_정류장현황.csv` | 3,639 rows (1,485 stops) | siting variable, route count |
| Parking lots | `data/raw/세종시_주차장_위치.csv` | 324 rows | siting variable |
| Child protection zones | `data/raw/세종시_어린이보호구역.csv` | 76 rows | map inspection |
| Eoulling (Sejong public bike) stations | `data/bicycle/세종_어울링/대여소현황_2022.csv` | 618 rows | per-district K |
| Eoulling trips | `이용현황_2024_1분기.csv`, `2분기.csv` | 1,229,686 rows | commute share |
| Per-district coordinates | `data/dong_coords/*.csv` | 21 districts, 8,400 rows total | clustering input |
| Seoul subway, bus stops, crosswalks, Ttareungyi stations | not in the repo | - | validation |

The raw speed files are per-road Excel sheets with a four-row header and interleaved "average travel speed" summary rows.

- Dropped the header, unified column names, derived the weekday, and removed every summary row (`code/03`).
- Excluded 01:00-05:00 so that near-empty night traffic would not inflate segment top speeds (`code/04`, `code/05`).
- Computed per-slot means by weekday, road, segment, hour and direction, and joined a per-segment top speed.

For bus stops, routes were counted per `정류장ID` and each stop's coordinate was repeated once per route, so that busier stops pull cluster centres harder (`code/22`, output `data/processed/새좌표.csv`, 10,969 rows). From building-permit data, only parking sites with more than 10 spaces were kept (`code/01`). The combined points were then clipped to district boundaries.

## 3. Method

### 3.1 Pipeline

```text
[Stage 1: case for e-bikes]
24 per-road speed .xls files
  → merge, drop summary rows, drop night hours       (code/03-05)
  → segment top speed = second-largest observation
  → per-slot deceleration → 0-4 level, summed over 4 slots (0-16)   (code/24, 25)
  → mean index per road/segment/direction → map     (code/26, 27)

[Stage 2: siting]
[validate] Seoul subway + bus stops + crosswalks
  → K-Means (k=2763) → distance from centres to Ttareungyi stations → share within 200 m   (code/13-15)

[apply] Sejong bus stops (route-weighted) + crosswalks + parking
  → split by osmnx district polygons                  (code/23, 32)
  → per-district K-Means (k = Eoulling stations in district, check silhouette > 0.5)
  ├─ MCLP coverage > 70%  → keep MCLP solution    (8 districts)
  └─ otherwise            → keep OPTICS solution  (6 districts)
```

- Stage 1 turns "do Sejong's roads actually slow down?" into a per-segment number.
- Validation checks whether centres derived from the same kind of variables coincide with stations of a long-running bike-share system.
- Application sets K per district and handles well-covered and poorly covered districts with different methods.

### 3.2 Segment top speed: the second-largest value

Using the raw maximum as the denominator lets one or two outlier readings define the baseline, which makes normal segments look heavily congested. The top speed was therefore the largest value strictly below the maximum.

`code/05_요일도로명구간시방향별 평균_최대값_차이 코드.py`
```python
def get_second_largest(group):
    result = {}
    for time_col in ['00분', '15분', '30분', '45분']:
        max_val = group[time_col].max()
        second_largest = group[time_col][group[time_col] < max_val].max()
        result[time_col] = second_largest
    return pd.Series(result)

second_largest_values = df2.groupby(['요일', '도로명', '구간', '방향']).apply(get_second_largest).reset_index()

final_df = pd.merge(day_time_mean_v, second_largest_values, on=['요일', '도로명', '구간', '방향'], suffixes=('_평균', '_최대값'))
```

Means are grouped down to the hour while the top speed is not, so every hourly mean for a given weekday, segment and direction is compared with the same top speed.

### 3.3 Road stress index

Deceleration is mapped to five levels (0-4), and the levels of the four 15-minute readings are summed into an hourly score of 0-16. In the first version (`code/24`), roads capped at 30 km/h showed low deceleration because they are slow by design, and were classified as uncongested. For cyclists these are roads that call for caution, so a rule was added: a top speed of 30 or less maps straight to level 4.

`code/25_도로 스트레스 지수_어린이보호구역 적용.py`
```python
def calculate_index(avg, max_val):
    if max_val <= 30:
        return 4
    ratio = (1 - (avg / max_val)) * 100
    if ratio <= 20:
        return 0
    elif 20 < ratio <= 25:
        return 1
    elif 25 < ratio <= 35:
        return 2
    elif 35 < ratio <= 45:
        return 3
    else:
        return 4
```

The index sum was then averaged per road, segment and direction (`code/26`) and drawn as coloured polylines (`code/27`). For the presentation, the five levels were collapsed into three deceleration bands: normal (0-20%), high (20-35%) and very high (35%+).

### 3.4 Proxy ground truth: Seoul's Ttareungyi

Seoul subway stations, bus stops and crosswalks were pooled, points outside Seoul's latitude/longitude bounds were dropped, and K-Means was fitted with `n_clusters=2763`, treating one centre as one station. Ttareungyi locations were used only for evaluation, never for fitting.

`code/14_k-means_정확도.py`
```python
# 클러스터 중심과 종속변수(따릉이 위치) 간의 거리 계산
distances_to_black = cdist(centroids, dependent_data[['lat', 'lng']])

# 각 클러스터 중심에서 가장 가까운 따릉이 위치까지의 거리 계산
min_distances_to_black = np.min(distances_to_black, axis=1)

# 클러스터 중심점 중에서 특정 거리 이내에 따릉이 위치가 있는 비율 계산 (약 200미터)
threshold_distance = 0.002
accuracy_centroid_within_threshold = np.mean(min_distances_to_black < threshold_distance)
```

A high share supports the claim that density centres of transit nodes explain where bike-share demand sits. Sejong has no subway, so subway stations were replaced with per-stop route counts and parking lots.

### 3.5 District split

Each district name was geocoded with `osmnx.geocode_to_gdf("<dong>, 세종시, South Korea")`, and every point was assigned with `shapely`'s `contains` test before writing one CSV per district (`code/32`). The list held 23 districts and towns; 21 files were written. Siting results were reported for 14 administrative districts.

### 3.6 Per-district K-Means, then MCLP or OPTICS

- **Choosing K.** To reflect differences in area and population, K was set to the district's current number of Eoulling stations. If the silhouette score was at least 0.5 that K was kept; otherwise the nearest K above 0.5 was searched for, and if none existed the district went to OPTICS.
- **MCLP.** A K-Means centre is only a mean position and says nothing about how much demand it covers, so MCLP was used to choose sites that cover as much demand as possible within a radius. The presentation's formulation:

```text
maximize   sum_i y_i
subject to sum_j x_j <= K                      # at most K docks (Eoulling stations in district)
           y_i <= sum_{j : d(i,j) <= R} x_j    # demand i is covered only if a dock lies within R
           x_j, y_i in {0, 1}
R: base radius 200 m (set for Saerom-dong, the median district by area), adjusted by density and area
```

- **OPTICS.** In low-coverage districts demand points are scattered, and a fixed-radius cover fits poorly. There, OPTICS was used to keep only dense regions as clusters, based on reachability distance `ReachDist(p,q) = max(CoreDist(p), Distance(p,q))`.
- **Decision rule.** `silhouette > 0.5 and coverage > 70%` kept the MCLP solution; anything else used OPTICS.

## 4. Experiments

### 4.1 Setup

| Item | Setting |
|---|---|
| Stress index scope | 5 weekdays, 00:00 and 06:00-23:00 (19 slots), 417 units |
| Deceleration level bounds | 20 / 25 / 35 / 45 %; top speed <= 30 maps to 4 |
| Seoul K-Means | `KMeans(n_clusters=2763, random_state=42)` on raw lat/lng |
| Proximity test | Euclidean distance in degrees `< 0.002`, taken as about 200 m |
| Sejong K per district | number of Eoulling stations, silhouette > 0.5 |
| MCLP radius | 200 m base, adjusted per district |
| Method split | coverage > 70% → MCLP, otherwise OPTICS |

### 4.2 Results

**Case for e-bikes.** The presentation reported 190 very-high, 140 high and 87 normal segments (417 in total), i.e. 79% at high or above. Excluding national and provincial roads, 153 segments (about 53.68%) averaged under 25 km/h at commute hours. The presentation took 25 km/h as the speed of a pedal-assist e-bike, so on those segments an e-bike can keep pace with a car. Of 1,229,686 Eoulling trips in Q1-Q2 2024, 18.26% were returned in commute windows (07:30-08:30, 18:00-19:00) (`code/33`, reproduced from the repo data).

**Per-district siting** (values from the presentation slides).

| District | K-Means silhouette | MCLP coverage | OPTICS silhouette | Chosen |
|---|---|---|---|---|
| Goun-dong | 0.56 | 71% | 0.70 | MCLP |
| Naseong-dong | 0.57 | 88% | 0.64 | MCLP |
| Dajeong-dong | 0.63 | 51% | 0.65 | OPTICS |
| Daepyeong-dong | 0.60 | 75% | 0.62 | MCLP |
| Dodam-dong | 0.63 | 51% | 0.63 | OPTICS |
| Bangok-dong | 0.65 | 72% | 0.68 | MCLP |
| Boram-dong | 0.59 | 54% | 0.64 | OPTICS |
| Saerom-dong | 0.53 | 38% | 0.71 | OPTICS |
| Sodam-dong | 0.61 | 21% | 0.61 | OPTICS |
| Areum-dong | 0.62 | 70% | 0.67 | MCLP |
| Eojin-dong | 0.68 | 71% | 0.72 | MCLP |
| Jongchon-dong | 0.52 | 59% | 0.72 | OPTICS |
| Hansol-dong | 0.78 | 87% | 0.73 | MCLP |
| Haemil-dong | 0.56 | 85% | 0.65 | MCLP |

**Ttareungyi validation.** The scoring code (`code/14`) is in the repo, but its output and the Seoul input files are not, so no proximity figure is given here.

### 4.3 Ablation

The stress-index rule was recomputed from the preprocessed file in the repo. Of the 417 units, 67 have a top speed of 30 or less in at least one slot.

| Variant | Mean of per-unit index | Median | Units with mean index >= 12 | Units < 4 |
|---|---|---|---|---|
| Deceleration only (`code/24`) | 8.58 | 9.94 | 100 | 73 |
| + top speed <= 30 → 4 (`code/25`, adopted) | 9.37 | 10.18 | 140 | 70 |

Effect of the method-selection rule:

| Rule | Districts on MCLP | Note |
|---|---|---|
| silhouette > 0.5 only | 14 | every district passes (lowest: Jongchon-dong, 0.52), so it does not discriminate |
| silhouette > 0.5 and coverage > 70% (adopted) | 8 | coverage spans 21-88% and decides the split |
| OPTICS silhouette | undecided | OPTICS also scores 0.61-0.73 in all 14 districts; silhouette alone cannot pick between methods |

Silhouette did not separate the districts; coverage did.

## 5. Limitations and future work

- **Distance approximation.** Treating 0.002 degrees as 200 m in Euclidean space makes the radius direction-dependent (at Seoul's latitude, 0.002 deg is about 220 m north-south and about 180 m east-west). Haversine distance or a metric projection should be used.
- **No random baseline.** The Ttareungyi proximity share was never compared with the same number of random points or a uniform grid. Stations are dense in central Seoul, so without a baseline the share cannot be called high or low.
- **Boundary case and post-hoc threshold.** Areum-dong shows exactly 70% coverage on the slide, on the edge of the `> 70%` rule, yet was assigned to MCLP. The 70% threshold was itself chosen after seeing results. Thresholds and tie handling should be fixed in advance.
- **Stress-index approximations.** The 30 km/h rule stands in for matching child-protection zones to segments. The second-largest value fails when there are two or more outliers; a quantile would be more robust. Re-aggregating the three bands from the repo file gives 219/133/65 rather than the presented 190/140/87, so the presentation's aggregation cannot be confirmed from code.
- **Reproducibility.** The Sejong per-district K-Means/MCLP/OPTICS code and the Seoul validation inputs are not in the repo, and scripts depend on local absolute paths. Per-district results come from the presentation.

## References

- Code: https://github.com/tjehdgus/sejong_kmeans
- Development log (Tistory, Korean): https://mynews76154.tistory.com/22
- Methods: K-Means, silhouette score, MCLP (Maximal Covering Location Problem, Church & ReVelle), OPTICS (Ankerst et al.)
- Libraries: scikit-learn, osmnx, shapely, folium
