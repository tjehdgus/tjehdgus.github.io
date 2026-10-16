---
title: "Classifying 396 Car Models: Noise Filtering, ConvNeXt V2 with EMA/SWA, and Temperature Scaling for LogLoss"
description: "A pipeline for classifying car images into 396 models in the Dacon HAI! Hecto AI Challenge. Test-distribution-aware noise filtering, a five-backbone comparison, ConvNeXt V2 base trained with EMA, SWA and curriculum learning, and temperature scaling, TTA and soft voting at inference reached a private LogLoss of 0.12452."
date: 2026-10-16
lang: en
key: car-classification
tags: [Image Classification, Fine-grained, ConvNeXt V2, EMA, SWA, Temperature Scaling, Dacon]
tistory: https://mynews76154.tistory.com/16
---

## Abstract

In June 2025 our three-person team entered the Dacon HAI! Hecto AI Challenge, where each car image had to be assigned to one of 396 car models. The metric was LogLoss, so probability calibration mattered as much as accuracy. We removed only 63 training images, using "does this kind of image appear in the test set" as the criterion, and picked ConvNeXt V2 base after comparing five backbones. Training combined Mixup/CutMix, EMA, SWA and curriculum learning; inference applied temperature scaling, TTA and soft voting. LogLoss went from 0.1718 for the single backbone to 0.11653 public and 0.12452 private, which placed 36th. The repository does not separate individual contributions, so the work is described at the team level.

## 1. Problem

- Input: one car image (jpg, organized as one folder per class)
- Output: a probability for each of the 396 car models
- Metric: multi-class LogLoss (lower is better)

```text
LogLoss = -(1/N) * sum_i sum_j y[i][j] * log(p[i][j])
# N: number of images, j: 396 classes, y: one-hot label, p: softmax probability
```

This is a fine-grained task with many classes and few images per class. Many classes differ only in local parts such as the grille, headlights or badge. LogLoss also heavily penalizes confident mistakes. We therefore treated training techniques that improve classification and inference techniques that shape the probabilities as two separate design problems.

## 2. Data

| Item | Value |
|---|---|
| Training set | 396 class folders, 33,137 images |
| Training set after filtering | 33,074 images (63 removed) |
| Test set | 8,258 images |
| Images per class | 57 to 88; about 70% of classes have 80 or more |
| Image width | about 80% between 500 and 900 px |
| Image height | about 85% between 400 and 600 px |

Class imbalance was mild, with a max/min ratio of about 1.5:1. Inputs were resized to squares (384 or 512).

### 2.1 Noise filtering

Filtering was done twice. `code/noise_filter.py` moved the listed files from the training folder into a separate noise folder, with `--dry-run` to preview the move.

| Version | Criterion | Removed | Training images |
|---|---|---|---|
| Ver.1 | Images judged harmful for training, picked by eye (damaged, open trunk, interior side view, strong reflections) | 497 | 32,640 |
| Ver.2 (final) | Images where the car is not fully visible and cannot be classified, or whose type does not appear in the test set (interior shots, unidentifiable) | 63 | 33,074 |

Ver.2 changed the question from "is this image clean" to "does this type of image appear in the test set". Open-trunk and reflection-heavy photos also occur in the test set, so they stayed.

### 2.2 Augmentation

The `autoaug` policy in `code/timm_auto_augment.py` applies timm's ImageNet AutoAugment (`original`) and RandomErasing (`re_prob=0.15`), preceded by a random choice among plain Resize, RandomResizedCrop (scale 0.8 to 1.0), and an upscale followed by CenterCrop.

Mixup/CutMix were applied per batch on the GPU (Section 3.3). Lighting augmentations such as RandomShadow and RandomSunFlare were implemented in `transforms_albu.py`. Mosaic was implemented but disabled with `USE_MOSAIC = False`: in classification, a single image containing pieces of four cars makes the label ambiguous.

## 3. Method

### 3.1 Pipeline overview

```text
Raw training data (33,137 images)
  └─ Noise filtering Ver.2 ─→ 33,074 images
       └─ Stratified hold-out (train 80% / val 20%, seed 42)
            ├─ Training: ConvNeXt V2 base (timm, 512px)
            │    ├─ AutoAugment + RandomErasing + Mixup/CutMix
            │    ├─ Curriculum learning (low-loss samples first, 50% → 100%)
            │    ├─ EMA (with warmup) + self-distillation (EMA teacher)
            │    └─ SWA (late-epoch weight averaging + BN recomputation)
            └─ Inference
                 ├─ Temperature scaling (grid search for T on part of val)
                 ├─ TTA (flip, ±3° rotation, scale, brightness ...)
                 └─ Soft-voting ensemble (inverse val-loss weights)
                      └─ Submission (396 class probabilities)
```

- Filtering: removed only image types absent from the test distribution, keeping as much data as possible.
- Validation: a single hold-out via `train_test_split(test_size=0.2, stratify=labels, random_state=42)`.
- Training: stacked techniques for accuracy and weight stability on a single backbone.
- Inference: reduced LogLoss by reshaping the probability distribution without changing the predicted class.

### 3.2 Backbone selection

Five timm pretrained backbones were compared on the leaderboard (Section 4.2), and ConvNeXt V2, a CNN, beat the four ViT-family models. The final training code uses `convnextv2_base.fcmae_ft_in22k_in1k_384`, a base-size model pretrained with FCMAE and fine-tuned on ImageNet-22k and 1k.

The hierarchical CNN's strength on local patterns fits cues like grilles, headlights and badges, and the base/large variants make the cost easy to adjust. ConvNeXt V2 was not clearly ahead in training time or stability; the deciding factor was the leaderboard score.

Training used AdamW, AMP and gradient checkpointing with batch size 64, configured for an RTX A6000. The scheduler was `WarmupCosineScheduler` from `schedulers.py` (warmup followed by cosine decay).

### 3.3 Mixup / CutMix

`code/gpu_mixup_module.py` decides per batch, with two separate probabilities, whether to mix and whether to use Mixup or CutMix. The settings were `FIXED_MIXUP_ALPHA = 0.5`, `FIXED_MIXUP_PROB = 0.3` and `mixup_ratio=0.5`. An adaptive controller that adjusts mixing strength from the degree of overfitting (`robust_adaptive_mixup_controller_v3.py`) was also implemented, but the final script used fixed values (`MIXUP_MODE = "fixed"`).

### 3.4 Curriculum learning

Per-sample losses from the previous epoch are recorded, and the next epoch trains on only a fraction of the samples, starting from the lowest-loss ones. The fraction grows linearly from `CURRICULUM_START_RATIO = 0.5` to `CURRICULUM_END_RATIO = 1.0` at the last epoch. Batches with Mixup applied are excluded from difficulty measurement, since their loss is computed against mixed labels.

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

### 3.5 EMA, self-distillation, SWA

**EMA.** `code/ema.py` keeps an exponential moving average of the weights as a separate model used for validation and inference. During the first `EMA_WARMUP_EPOCHS`, the decay starts small as `min(decay, (1 + updates) / (10 + updates))` and grows. Every 3 epochs the EMA and raw models were compared on validation.

**Self-distillation.** `train_acc_improved_timm_curr.py` uses the EMA model as a teacher and trains on `(1 - α) * base_loss + α * distill_loss` (`DISTILLATION_ALPHA = 0.2`, `DISTILLATION_TEMPERATURE = 4.0`, starting at epoch 10).

**SWA.** `SWAManager` in `code/swa_manager.py` wraps PyTorch's `AveragedModel` and `SWALR`, and recomputes BatchNorm statistics with `update_bn` after averaging. The script above starts SWA at epoch 35 with `SWA_LR = 5e-6`. The slides record SWA's role as narrowing the gap between CV and leaderboard scores.

### 3.6 Temperature scaling

Dividing the logits by a temperature T changes only how sharp the softmax is; the argmax, and therefore accuracy, is unchanged. `auto_temperature_tuning_fixed.py` takes up to 3,000 validation images and splits them in half for tuning and evaluation. It collects the tuning logits once and grid-searches 30 values of T in [0.09, 1.5] for the lowest cross entropy.

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

Training techniques like Mixup make the model less confident, so the optimal T can fall below 1; the lower bound was set to 0.09 for that reason. The `_fixed.py` variant adds `fix_swa_state_dict` to handle SWA checkpoints.

### 3.7 TTA and soft voting

`predict_improved_unified.py` applies temperature scaling together with TTA and combines TTA outputs with a geometric mean by default. `adaptive_tta.py` applies transforms in order of importance (original, horizontal flip, ±3° rotation, scale, brightness, ...) and stops early once a confidence score of `max_prob * 0.6 + (top1 - top2) * 0.4` exceeds 0.85. Finally, `ensemble_weighted_auto.py` performs soft voting over per-class probabilities from several models, weighting each model by its normalized inverse validation loss (`inverse_loss`).

## 4. Experiments

### 4.1 Setup

- Validation: a single stratified 80/20 hold-out of the filtered training set (seed 42). Temperature tuning split part of that validation set in half again.
- Hardware: settings targeted an RTX A6000 (per comments in the training script).
- The training script exists in several variants (`_curr`, `_swa`, `_best`, ...) with slightly different values. The table lists the final configuration from the slides next to a representative script.

| Item | Slides (final model) | `train_acc_improved_timm_curr.py` |
|---|---|---|
| Model | ConvNeXt V2 | `convnextv2_base.fcmae_ft_in22k_in1k_384` |
| Batch / epochs | 64 / 50 | 64 / 50 |
| Image size | 384 / 512 | 512 |
| Learning rate / weight decay | 5e-5 / 0.05 | 5e-5 / 0.05 (AdamW) |
| Warmup | - | 10 epochs |
| Label smoothing | 0.05 | 0 |
| EMA decay / warmup | 0.999 / 5 epochs | 0.9995 / 5 epochs |
| Mixup / CutMix | on | alpha 0.5, prob 0.3, ratio 0.5 |
| SWA | on | from epoch 35, lr 5e-6 |
| Early stopping patience | 10 | 20 |

The exact configuration of the final submission cannot be pinned down from the repository alone.

### 4.2 Results

Backbone comparison (leaderboard LogLoss, from the slides):

| Backbone | Family | LB |
|---|---|---|
| Swin Transformer | Hierarchical ViT | 0.2259 |
| BEiT v2 | Self-supervised ViT | 0.2166 |
| EVA-02 | MIM-pretrained ViT | 0.2002 |
| DeiT v3 | Distillation-based ViT | 0.1799 |
| **ConvNeXt V2** | CNN | **0.1718** |

Final result:

| Configuration | Public LB | Private LB |
|---|---|---|
| ConvNeXt V2, single backbone | 0.1718 | - |
| Final submission (training techniques + temperature scaling + TTA + soft voting) | 0.11653 | 0.12452 (36th) |

Public LogLoss dropped by about 0.055 relative to the single backbone, and the private score was about 0.008 higher than the public one. The competition had 748 participating teams (reported figure).

### 4.3 Ablation

| Variant | Compared with | Result | Decision |
|---|---|---|---|
| Warmup + CosineAnnealing | ReduceLROnPlateau | LB 0.1558 vs 0.1575 | Warmup + cosine adopted |
| Noise filtering Ver.1 (497 removed) | Ver.2 (63 removed) | no score recorded | Ver.2 adopted |
| Mosaic (p=0.3) | none | no score recorded | dropped (ambiguous labels) |
| Adaptive Mixup controller | fixed alpha/prob | no score recorded | fixed values adopted |
| EMA + SWA / SWA only | each other | no score recorded | both scripts kept |
| Curriculum learning | none | no score recorded | used |

Apart from the scheduler, the repository holds no per-component scores, so the improvement from 0.1718 to 0.11653 cannot be attributed to individual techniques.

## 5. Limitations and future work

- **Single hold-out validation.** Holding out 20% of 57 to 88 images per class leaves roughly 11 to 18 validation images per class, so validation loss is noisy. K-fold OOF predictions would give more reliable temperatures and ensemble weights.
- **Leaderboard-driven choices.** Backbone and scheduler were chosen on the leaderboard. The private score being about 0.008 worse than public suggests some decisions fit the public split.
- **No contribution log.** Scores for individual variants were not kept. Each experiment needs a one-line record of its settings and score.
- **Inconsistent settings across variants.** Label smoothing, EMA decay and patience differ between the slides and the scripts. The final configuration should live in a single config file.
- **Interaction between filtering and curriculum.** Ambiguous samples that were kept are treated as hard samples by the curriculum. Separating the two with loss-based noise detection remains to be tested.

## References

- Code: https://github.com/tjehdgus/dacon_car_convnext_v2
- Development log (Tistory, Korean): https://mynews76154.tistory.com/16
- Models and methods: ConvNeXt V2 (FCMAE), Swin Transformer, EVA-02, DeiT III, BEiT v2, AutoAugment, Mixup, CutMix, Stochastic Weight Averaging, Temperature Scaling (On Calibration of Modern Neural Networks), timm
