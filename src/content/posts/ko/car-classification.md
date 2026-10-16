---
title: "396종 차량 이미지 분류: 노이즈 정제, ConvNeXt V2 + EMA·SWA, Temperature Scaling으로 LogLoss 줄이기"
description: "데이콘 HAI! Hecto AI Challenge에서 차량 이미지를 396개 차종으로 분류한 파이프라인을 정리했다. 평가 분포 기준의 노이즈 정제, 백본 5종 비교, ConvNeXt V2 base 위의 EMA·SWA·Curriculum 학습, Temperature Scaling·TTA·soft voting 추론으로 Private LogLoss 0.12452를 기록했다."
date: 2026-10-16
lang: ko
key: car-classification
tags: [Image Classification, Fine-grained, ConvNeXt V2, EMA, SWA, Temperature Scaling, Dacon]
tistory: https://mynews76154.tistory.com/16
---

## 초록

2025년 6월 데이콘 HAI! Hecto AI Challenge에 3인 팀으로 참가해 차량 이미지 한 장을 396개 차종 중 하나로 분류했다. 평가 지표가 LogLoss라서 정확도보다 예측 확률의 보정이 점수를 좌우했다. 학습 데이터는 평가 데이터에 없는 유형 기준으로 63장만 제거했고, 백본 5종을 비교해 ConvNeXt V2 base를 골랐다. 학습에는 Mixup/CutMix, EMA, SWA, Curriculum Learning을, 추론에는 Temperature Scaling, TTA, soft voting을 적용했다. 단일 백본 LB 0.1718에서 최종 Public 0.11653, Private 0.12452로 내려갔고 36위를 기록했다. 팀원별 작업은 레포로 구분되지 않아 팀 단위로 서술한다.

## 1. 문제 정의

- 입력: 차량 이미지 1장 (jpg, 클래스별 폴더 구조)
- 출력: 396개 차종 각각에 대한 확률
- 평가 지표: 다중 클래스 LogLoss (낮을수록 좋음)

```text
LogLoss = -(1/N) * sum_i sum_j y[i][j] * log(p[i][j])
# N: 이미지 수, j: 396개 클래스, y: one-hot 정답, p: softmax 확률
```

클래스는 많고 클래스당 장수는 적은 세분류(fine-grained) 문제로, 그릴, 헤드라이트, 로고 같은 국소 부위로 갈리는 클래스가 많다. 또 LogLoss는 확신을 갖고 틀린 예측에 큰 벌점을 준다. 그래서 분류 성능을 올리는 학습 기법과 확률을 다듬는 추론 기법을 따로 설계했다.

## 2. 데이터

| 항목 | 값 |
|---|---|
| 학습 데이터 | 396개 클래스 폴더, 33,137장 |
| 정제 후 학습 데이터 | 33,074장 (63장 제거) |
| 평가 데이터 | 8,258장 |
| 클래스당 이미지 수 | 57~88장, 전체 클래스의 약 70%가 80장 이상 |
| 이미지 너비 | 500~900px 구간에 약 80% |
| 이미지 높이 | 400~600px 구간에 약 85% |

클래스 불균형은 최대/최소 약 1.5:1로 심하지 않았다. 입력은 정사각형(384 또는 512)으로 리사이즈했다.

### 2.1 노이즈 정제

정제는 두 번 했다. 대상 파일은 `code/noise_filter.py`로 학습 폴더에서 노이즈 폴더로 옮겼고, `--dry-run`으로 먼저 확인했다.

| 버전 | 기준 | 제거 | 학습 데이터 |
|---|---|---|---|
| Ver.1 | 학습에 방해된다고 판단한 이미지를 육안으로 선별 (훼손, 트렁크 열림, 측면 내부, 과도한 비침) | 497장 | 32,640장 |
| Ver.2 (최종) | 차량이 온전히 보이지 않아 분류가 불가능하거나, 평가 데이터에 유사한 유형이 없는 이미지 (차량 내부, 식별 불가) | 63장 | 33,074장 |

Ver.2는 기준을 "깨끗한가"에서 "평가 데이터에 나오는 유형인가"로 바꿨다. 트렁크 열림이나 반사 사진은 평가 데이터에도 있어 남겼다.

### 2.2 증강

`code/timm_auto_augment.py`의 `autoaug` 정책은 timm ImageNet AutoAugment(`original`)와 RandomErasing(`re_prob=0.15`)을 쓰고, 앞단에서 Resize, RandomResizedCrop(scale 0.8~1.0), 확대 후 CenterCrop 중 하나를 무작위로 고른다.

Mixup/CutMix는 배치 단위로 GPU에서 적용했다(3.3절). RandomShadow, RandomSunFlare 같은 조명 증강은 `transforms_albu.py`에 구현했고, Mosaic은 구현했지만 `USE_MOSAIC = False`로 껐다. 분류에서는 한 이미지에 네 차량의 조각이 섞여 라벨이 애매해지기 때문이다.

## 3. 방법

### 3.1 전체 파이프라인

```text
원본 학습 데이터 (33,137장)
  └─ 노이즈 정제 Ver.2 ─→ 33,074장
       └─ Stratified hold-out (train 80% / val 20%, seed 42)
            ├─ 학습: ConvNeXt V2 base (timm, 512px)
            │    ├─ AutoAugment + RandomErasing + Mixup/CutMix
            │    ├─ Curriculum Learning (loss 낮은 샘플부터 50% → 100%)
            │    ├─ EMA (warmup 포함) + Self-Distillation (EMA teacher)
            │    └─ SWA (후반 epoch 가중치 평균 + BN 재계산)
            └─ 추론
                 ├─ Temperature Scaling (val 일부로 T 격자 탐색)
                 ├─ TTA (flip, ±3도 회전, scale, brightness ...)
                 └─ Soft voting 앙상블 (val loss 역수 가중치)
                      └─ 제출 (396 클래스 확률)
```

- 정제: 평가 분포에 없는 유형만 제거해 학습 데이터를 보존했다.
- 검증: `train_test_split(test_size=0.2, stratify=labels, random_state=42)` 단일 홀드아웃.
- 학습: 분류 성능과 가중치 안정성을 올리는 기법을 한 백본 위에 누적했다.
- 추론: 정확도는 두고 확률 분포만 조정해 LogLoss를 줄였다.

### 3.2 백본 선택

timm 사전학습 백본 5종을 리더보드로 비교했고(4.2절), CNN인 ConvNeXt V2가 ViT 계열 4종보다 낮았다. 최종 학습 코드의 모델은 `convnextv2_base.fcmae_ft_in22k_in1k_384`로, FCMAE로 사전학습하고 ImageNet-22k, 1k로 파인튜닝한 base 모델이다.

계층형 CNN의 지역 패턴 학습이 그릴, 헤드라이트, 로고 같은 국소 단서에 맞고, base/large 크기 선택지로 비용을 조절하기 쉽다는 점을 함께 고려했다. 학습 시간과 안정성은 다른 백본보다 뚜렷하게 앞서지 않았고, 최종 근거는 리더보드 점수였다.

학습은 AdamW, AMP, gradient checkpointing을 켜고 RTX A6000 기준 batch 64로 돌렸다. 스케줄러는 `schedulers.py`의 `WarmupCosineScheduler`(warmup 후 cosine 감소)를 썼다.

### 3.3 Mixup / CutMix

`code/gpu_mixup_module.py`는 배치마다 섞기 적용 여부와 Mixup/CutMix 선택을 각각 확률로 정한다. 설정은 `FIXED_MIXUP_ALPHA = 0.5`, `FIXED_MIXUP_PROB = 0.3`, `mixup_ratio=0.5`다. 과적합 정도에 따라 강도를 바꾸는 적응형 컨트롤러(`robust_adaptive_mixup_controller_v3.py`)도 구현했지만 최종 스크립트는 고정값(`MIXUP_MODE = "fixed"`)을 썼다.

### 3.4 Curriculum Learning

이전 epoch의 샘플별 loss를 기록해 두고, 다음 epoch에는 loss가 낮은 샘플부터 일정 비율만 학습한다. 비율은 `CURRICULUM_START_RATIO = 0.5`에서 시작해 마지막 epoch에 `CURRICULUM_END_RATIO = 1.0`이 되도록 선형으로 늘린다. Mixup이 적용된 배치는 섞인 라벨 기준의 loss라 난이도 측정에서 뺐다.

`code/train_acc_improved_timm_curr.py`
```python
def get_curriculum_samples(all_indices, epoch, prev_losses):
    """Curriculum Learning을 위한 샘플 선택"""
    # 현재 에폭의 샘플 비율 계산
    progress = (epoch - 1) / (N_EPOCHS - 1)  # 0~1
    current_ratio = CURRICULUM_START_RATIO + (CURRICULUM_END_RATIO - CURRICULUM_START_RATIO) * progress

    # 사용할 샘플 수 계산
    n_samples = int(len(all_indices) * current_ratio)

    # loss 낮은 순서로 정렬하여 쉬운 샘플부터 선택
    if prev_losses is not None and len(prev_losses) == len(all_indices):
        sorted_indices = np.argsort(prev_losses)[:n_samples]
        return [all_indices[i] for i in sorted_indices]
    else:
        # 첫 에폭이거나 loss 정보가 없으면 전체 사용
        return all_indices
```

### 3.5 EMA, Self-Distillation, SWA

**EMA.** `code/ema.py`가 가중치의 지수이동평균을 별도 모델로 유지해 검증과 추론에 썼다. 초반 `EMA_WARMUP_EPOCHS` 동안은 decay를 `min(decay, (1 + updates) / (10 + updates))`로 작게 시작해 키우고, 3 epoch마다 EMA와 원본 모델의 검증 점수를 비교했다.

**Self-Distillation.** `train_acc_improved_timm_curr.py`는 EMA 모델을 teacher로 삼아 `(1 - α) * base_loss + α * distill_loss`로 학습한다(`DISTILLATION_ALPHA = 0.2`, `DISTILLATION_TEMPERATURE = 4.0`, 10 epoch부터 적용).

**SWA.** `code/swa_manager.py`의 `SWAManager`가 PyTorch `AveragedModel`과 `SWALR`을 감싸고, 평균 후 `update_bn`으로 BatchNorm 통계를 다시 계산한다. 위 스크립트는 35 epoch부터 SWA를 시작하고 `SWA_LR = 5e-6`을 쓴다. 발표 자료에 기록한 SWA의 역할은 CV와 LB 간 점수 격차 완화다.

### 3.6 Temperature Scaling

logit을 온도 T로 나눠 softmax의 날카로움만 조정한다. argmax가 바뀌지 않으므로 정확도는 그대로다. `auto_temperature_tuning_fixed.py`는 검증 셋에서 최대 3,000장을 떼어 절반은 튜닝, 절반은 평가에 쓴다. 튜닝용 logit을 한 번 모은 뒤 0.09~1.5 구간을 30개 격자로 훑어 cross entropy가 가장 낮은 T를 고른다.

`code/auto_temperature_tuning_fixed.py`
```python
temperatures = np.linspace(temp_range[0], temp_range[1], num_temps)
best_temp = 1.0
best_loss = float('inf')

losses = []
for temp in tqdm(temperatures, desc="Testing temperatures"):
    scaled_logits = all_logits / temp
    loss = F.cross_entropy(scaled_logits, all_labels).item()
    losses.append(loss)

    if loss < best_loss:
        best_loss = loss
        best_temp = temp
```

Mixup처럼 덜 확신하게 만드는 학습 기법을 쓰면 최적 T가 1보다 작게 나올 수 있어 탐색 하한을 0.09로 낮게 뒀다. `_fixed.py`는 SWA 체크포인트용 `fix_swa_state_dict`를 추가한 버전이다.

### 3.7 TTA와 soft voting

`predict_improved_unified.py`는 온도 보정과 TTA를 함께 적용하고 TTA 결과를 geometric 평균(기본)으로 합친다. `adaptive_tta.py`는 원본, 좌우 반전, ±3도 회전, scale, brightness 등 변환을 중요도 순으로 적용하다가 `max_prob * 0.6 + (top1 - top2) * 0.4`로 계산한 신뢰도가 0.85를 넘으면 남은 변환을 생략한다. 마지막으로 `ensemble_weighted_auto.py`가 여러 모델의 클래스별 확률을 평균하는 soft voting을 하며, 기본 가중치는 각 모델 val loss의 역수(`inverse_loss`)를 정규화한 값이다.

## 4. 실험

### 4.1 실험 설정

- 검증: 정제 후 학습 데이터의 stratified 80/20 단일 홀드아웃(seed 42). Temperature 튜닝은 그 검증 셋의 일부를 다시 반으로 나눠 썼다.
- 하드웨어: RTX A6000 기준 설정(학습 스크립트 주석).
- 학습 스크립트가 `_curr`, `_swa`, `_best` 등 여러 변형으로 나뉘어 값이 조금씩 다르다. 아래는 발표 자료의 최종 설정과 대표 스크립트 값이다.

| 항목 | 발표 자료 (최종 모델) | `train_acc_improved_timm_curr.py` |
|---|---|---|
| 모델 | ConvNeXt V2 | `convnextv2_base.fcmae_ft_in22k_in1k_384` |
| batch / epochs | 64 / 50 | 64 / 50 |
| image size | 384 / 512 | 512 |
| learning rate / weight decay | 5e-5 / 0.05 | 5e-5 / 0.05 (AdamW) |
| warmup | - | 10 epoch |
| label smoothing | 0.05 | 0 |
| EMA decay / warmup | 0.999 / 5 epoch | 0.9995 / 5 epoch |
| Mixup / CutMix | 사용 | alpha 0.5, prob 0.3, ratio 0.5 |
| SWA | 사용 | 35 epoch부터, lr 5e-6 |
| early stopping patience | 10 | 20 |

최종 제출의 정확한 설정 조합은 레포만으로 특정되지 않는다.

### 4.2 결과

백본 비교 (리더보드 LogLoss, 발표 자료 기준):

| 백본 | 계열 | LB |
|---|---|---|
| Swin Transformer | 계층형 ViT | 0.2259 |
| BEiT v2 | 자기지도 ViT | 0.2166 |
| EVA-02 | MIM 사전학습 ViT | 0.2002 |
| DeiT v3 | 증류 기반 ViT | 0.1799 |
| **ConvNeXt V2** | CNN | **0.1718** |

최종 결과:

| 구성 | Public LB | Private LB |
|---|---|---|
| ConvNeXt V2 단일 백본 | 0.1718 | - |
| 최종 제출 (학습 기법 + Temperature Scaling + TTA + soft voting) | 0.11653 | 0.12452 (36위) |

단일 백본 대비 Public 기준 약 0.055 낮아졌고, Private는 Public보다 약 0.008 높았다. 참가 팀 수는 대회 결과 기준 748팀이다(보고된 값).

### 4.3 비교 실험

| 변형 | 비교 대상 | 결과 | 처리 |
|---|---|---|---|
| Warmup + CosineAnnealing | ReduceLROnPlateau | LB 0.1558 vs 0.1575 | Warmup + Cosine 채택 |
| 노이즈 정제 Ver.1 (497장 제거) | Ver.2 (63장 제거) | 점수 기록 없음 | Ver.2 채택 |
| Mosaic (p=0.3) | 미사용 | 점수 기록 없음 | 제외 (분류 라벨이 애매해짐) |
| 적응형 Mixup 컨트롤러 | 고정 alpha/prob | 점수 기록 없음 | 고정값 채택 |
| EMA + SWA / SWA 단독 | 상호 비교 | 점수 기록 없음 | 스크립트 양쪽 보존 |
| Curriculum Learning | 미사용 | 점수 기록 없음 | 사용 |

스케줄러 외에는 구성 요소별 점수가 레포에 없어, 0.1718에서 0.11653까지의 개선을 기법별로 분해할 수 없다.

## 5. 한계와 향후 과제

- **단일 홀드아웃 검증.** 클래스당 57~88장에서 20%를 떼면 검증 클래스당 11~18장 수준이라 val loss의 분산이 크다. K-fold OOF를 만들면 온도와 앙상블 가중치를 더 안정적으로 정할 수 있다.
- **리더보드 기반 선택.** 백본과 스케줄러를 리더보드로 골랐다. Private가 Public보다 약 0.008 높은 것은 Public에 맞춰진 판단이 있었을 가능성을 보여 준다.
- **기여도 기록 부재.** 변형별 점수가 남아 있지 않다. 실험마다 설정과 점수를 한 줄씩 남기는 로그가 필요하다.
- **변형 간 설정 불일치.** 발표 자료와 스크립트의 label smoothing, EMA decay, patience 값이 다르다. 최종 설정을 설정 파일 하나로 고정해야 한다.
- **정제와 커리큘럼의 상호작용.** 남겨 둔 애매한 샘플도 커리큘럼에서는 어려운 샘플로 취급된다. loss 기반 노이즈 탐지로 둘을 분리하는 실험이 남아 있다.

## 참고

- 코드: https://github.com/tjehdgus/dacon_car_convnext_v2
- 개발 기록 (티스토리): https://mynews76154.tistory.com/16
- 모델·기법: ConvNeXt V2 (FCMAE), Swin Transformer, EVA-02, DeiT III, BEiT v2, AutoAugment, Mixup, CutMix, Stochastic Weight Averaging, Temperature Scaling (On Calibration of Modern Neural Networks), timm
