---
title: "운동 자세 분류·평가: AI Hub 관절 좌표 기반 TSM-BiLSTM 분류기와 MediaPipe 규칙 평가"
description: "AI Hub 피트니스 자세 데이터의 2D·3D 관절 좌표로 5종 운동을 분류하는 TSM-Conv1D-BiLSTM 모델을 학습하고, MediaPipe Pose로 추출한 영상 관절에 규칙 기반 자세 조건 평가를 붙여 Perfect/Good/Bad 판정을 출력했다. 검증 3,060프레임에서 정확도 0.95, macro F1 0.93을 기록했다."
date: 2026-03-05
lang: ko
key: exercise-posture
tags: [Pose Estimation, MediaPipe, TSM, BiLSTM, PyTorch, Action Recognition]
---

## 초록

2024년 교육 과정 팀 프로젝트로, 운동 영상에서 운동 종류를 분류하고 자세를 평가하는 시스템을 만들었다. 학습 데이터는 AI Hub 피트니스 자세 데이터의 관절 좌표 JSON이며, 프레임마다 여러 view의 2D 좌표 평균과 3D 좌표를 이어 붙여 100차원 벡터를 만들었다. 분류기는 Temporal Shift 레이어, Conv1D, BatchNorm, 양방향 LSTM, FC 2층으로 구성했고 5종 운동을 분류한다. 추론 단계에서는 MediaPipe Pose로 영상의 관절을 추출해 분류기로 운동명을 정하고, 운동별 자세 조건을 규칙 함수로 판정해 Perfect/Good/Bad 단계를 화면에 표시했다. 프레임 단위 무작위 20% 검증셋 3,060개에서 정확도 0.95, macro F1 0.93, weighted F1 0.95를 기록했다. 레포는 본인이 업로드하고 정리했지만 커밋 이력으로 팀원별 작업을 구분할 수 없어 팀 단위로 서술한다.

## 1. 문제 정의

- 입력(학습): 한 프레임의 관절 좌표 벡터 (2D 20개 관절 x, y + 3D 20개 관절 x, y, z)
- 입력(추론): 운동 영상 (mp4). MediaPipe Pose로 프레임별 관절 좌표를 추출
- 출력 1: 5종 운동 중 하나 (버피 테스트, 스탠딩 니업, 스탠딩 사이드 크런치, 스텝 백워드 다이나믹 런지, 스텝 포워드 다이나믹 런지)
- 출력 2: 운동별 자세 조건의 O/X와 단계 판정 (Perfect / Good / Bad)
- 평가 지표: 분류는 클래스별 Precision, Recall, F1, 전체 Accuracy, macro/weighted 평균, 클래스별 ROC 곡선. 자세 평가는 정량 평가 없이 시연 영상으로 확인했다.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
macro avg = 클래스별 값의 단순 평균, weighted avg = 클래스별 샘플 수 가중 평균
```

제약 조건은 자원이었다. 발표 자료는 원본 데이터가 1TB를 넘어 메모리 부족으로 한 번에 학습할 수 없었다고 기록한다. 실제 학습 노트북은 AI Hub 데이터 중 하루 촬영분 폴더 하나(`Day13_201013_F`)의 라벨링 JSON만 사용한다.

## 2. 데이터

| 항목 | 값 |
|---|---|
| 출처 | AI Hub 피트니스 자세 이미지 데이터 (라벨링 JSON) |
| 사용 범위 | 촬영일 폴더 1개의 2D JSON과 대응하는 `-3d.json` |
| 샘플 단위 | 프레임 1장 = 샘플 1개 |
| 클래스 | 5종 운동 (`type_info.exercise`) |
| 피처 | 20개 관절 x 2D(x, y) = 40 + 20개 관절 x 3D(x, y, z) = 60, 합계 100 |
| 분할 | `train_test_split(test_size=0.2, random_state=42)`, 층화 없음 |
| 검증셋 | 3,060 프레임 |

검증셋 클래스 분포는 다음과 같다(`code/deep2.ipynb` 분류 리포트의 support).

| 클래스 | 검증 프레임 수 |
|---|---|
| 버피 테스트 | 824 |
| 스탠딩 니업 | 413 |
| 스탠딩 사이드 크런치 | 849 |
| 스텝 백워드 다이나믹 런지 | 169 |
| 스텝 포워드 다이나믹 런지 | 805 |

JSON에는 운동명 외에 3~5개의 자세 조건과 그 충족 여부가 들어 있다(발표 자료 기준). 분류 모델은 운동명만 라벨로 썼다.

### 2.1 2D·3D 좌표 결합

2D 데이터는 한 프레임을 여러 카메라 view로 찍어 view마다 관절 좌표가 있고, 3D 데이터는 view 없이 관절당 (x, y, z) 하나다. 두 형식을 한 벡터로 합치기 위해 2D는 view 평균을 내고, 같은 프레임 인덱스의 3D 좌표를 뒤에 이어 붙였다. 관절 목록은 Nose, 눈, 귀, 어깨, 팔꿈치, 손목, 골반, 무릎, 발목 좌우와 Neck, Back, Waist의 20개다.

`code/deep2.ipynb`

```python
for frame_idx, frame_data in enumerate(data_2d["frames"]):
    frame_coords = []

    # 2D 관절 좌표
    for joint in joint_names:
        view_coords = [frame_data[view]["pts"].get(joint, {"x": 0, "y": 0}) for view in frame_data if "view" in view]
        avg_x = np.mean([coord["x"] for coord in view_coords])
        avg_y = np.mean([coord["y"] for coord in view_coords])
        frame_coords.extend([avg_x, avg_y])

    # 3D 데이터 처리 (존재하는 경우)
    if data_3d and frame_idx < len(data_3d["frames"]):
        for joint in joint_names:
            coord_3d = data_3d["frames"][frame_idx]["pts"].get(joint, {"x": 0, "y": 0, "z": 0})
            frame_coords.extend([coord_3d["x"], coord_3d["y"], coord_3d["z"]])

    X_data.append(frame_coords)
    y_labels.append(data_2d['type_info'].get('exercise', 'Unknown'))
```

3D 파일이 없는 프레임은 가장 긴 벡터 길이에 맞춰 0으로 패딩했고, 전체 값을 데이터셋 최댓값으로 나눠 정규화했다.

## 3. 방법

### 3.1 전체 파이프라인

```text
[학습]
AI Hub 라벨링 JSON (2D 다중 view + 3D)
  └─ 프레임별 좌표 결합 (2D view 평균 40 + 3D 60 = 100차원)
       └─ 0 패딩 + 최댓값 정규화
            └─ 80/20 무작위 분할 (seed 42)
                 └─ 채널 축으로 120회 복제 → (B, 120, 100)
                      └─ TSM-Conv1D-BiLSTM 분류기 (5 클래스)
                           └─ EarlyStopping(patience 20) → 전체 모델 저장 (.pth)

[추론]
운동 영상 (mp4)
  ├─ 1차 패스: MediaPipe Pose (model_complexity=2)
  │    ├─ 관절 추출 + Neck/Pelvis/Spine 계산 → 120차원으로 패딩
  │    ├─ 지수 평활 (alpha 0.8) + [-1, 1] min-max 정규화
  │    ├─ 연속 2프레임을 분류기에 입력
  │    └─ 전체 예측의 최빈값 = 영상의 운동명
  └─ 2차 패스: 프레임별 자세 평가
       ├─ 운동명으로 자세 조건 목록 조회
       ├─ 조건별 규칙 함수로 O/X 판정
       └─ 충족 개수로 Perfect / Good / Bad 표시
```

- 좌표 결합: 다중 view 2D와 단일 view 3D를 같은 길이의 벡터로 맞췄다.
- 분류기: 시간 이동 레이어와 Conv1D로 국소 패턴을, BiLSTM으로 순서 정보를 학습하도록 설계했다.
- 운동명 결정: 프레임별 예측이 흔들리므로 영상 전체 예측의 최빈값 하나로 고정했다.
- 자세 평가: 학습 모델이 아니라 관절 좌표의 거리, 각도, 정렬을 보는 규칙 함수로 판정했다.

### 3.2 분류 모델

`code/deep2.ipynb`

```python
class TSMLayer(nn.Module):
    def forward(self, x):
        N, C, T = x.size()  # (batch_size, channels, time_steps)
        shift_left = x[:, :, :-self.shift_size]
        shift_right = x[:, :, self.shift_size:]
        shifted = torch.cat([shift_left, shift_right], dim=2)
        return shifted

class TSMModel(nn.Module):
    def forward(self, x):
        x = self.tsm(x)  # Apply Temporal Shift
        x = self.conv1(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)  # (batch, sequence_length, input_size)
        x, _ = self.lstm1(x)
        x = self.dropout(x[:, -1, :])  # 마지막 LSTM의 출력
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

코드 기준 텐서 흐름은 다음과 같다. 구현된 TSM 레이어는 원 논문처럼 채널 일부를 시간축으로 밀어 넣는 방식이 아니라, 한 칸 앞과 한 칸 뒤로 자른 두 텐서를 시간축으로 이어 붙인다. 그래서 길이가 `T`에서 `2T-2`로 늘어난다.

| 레이어 | 설정 | 출력 형태 (B=32) |
|---|---|---|
| 입력 | 100차원 벡터를 채널 축으로 120회 복제 | (32, 120, 100) |
| TSMLayer | shift_size=1, 두 슬라이스 연결 | (32, 120, 198) |
| Conv1d | 120 → 64, kernel 3, padding 1 | (32, 64, 198) |
| MaxPool1d + BatchNorm1d | pool 2 | (32, 64, 99) |
| BiLSTM | hidden 128, 양방향, 마지막 스텝 | (32, 256) |
| Dropout + FC + ReLU | p=0.2, 256 → 64 | (32, 64) |
| FC | 64 → 5 | (32, 5) |

학습 파라미터 수는 코드의 레이어 정의로 계산하면 238,661개다(BiLSTM 198,656, Conv1d 23,104, FC 16,773, BatchNorm 128). 저장된 `model/tsm_model_full.pth`의 텐서 크기(Conv1d 가중치 64x120x3, 최종 FC 5x64)도 이 정의와 맞는다. 발표 자료의 레이어 표는 Conv 이후 길이를 100, 50으로 적었지만 위 코드대로면 198, 99다.

### 3.3 MediaPipe 관절 추출과 보조 관절

MediaPipe Pose는 `static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5`로 설정했다. AI Hub 라벨에는 있지만 MediaPipe에는 없는 Neck, Pelvis, Spine은 랜드마크 평균으로 만들었다. 눈과 귀 4개는 매핑에서 `None`으로 두고 (0, 0, 0)을 넣었다.

`code/deep2.ipynb`

```python
    # Neck, Pelvis, Spine 계산 (수식)
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    neck_x = (left_shoulder.x + right_shoulder.x) / 2
    neck_y = (left_shoulder.y + right_shoulder.y) / 2
    neck_z = (left_shoulder.z + right_shoulder.z) / 2
    keypoints.extend([neck_x, neck_y, neck_z])

    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    pelvis_x = (left_hip.x + right_hip.x) / 2
    pelvis_y = (left_hip.y + right_hip.y) / 2
    pelvis_z = (left_hip.z + right_hip.z) / 2
    keypoints.extend([pelvis_x, pelvis_y, pelvis_z])

    spine_x = (pelvis_x + neck_x) / 2
    spine_y = (pelvis_y + neck_y) / 2
    spine_z = (pelvis_z + neck_z) / 2
    keypoints.extend([spine_x, spine_y, spine_z])
```

21개 매핑 관절과 보조 관절 3개를 (x, y, z)로 펼친 72개 값은 0을 채워 120개로 맞췄다. 이후 직전 프레임과 `0.8 * prev + 0.2 * new`로 평활하고, 프레임 안에서 [-1, 1]로 min-max 정규화했다. 분류기에는 연속 2프레임을 `(1, 120, 2)` 형태로 넣고, 영상 전체 예측의 최빈값을 운동명으로 썼다.

### 3.4 규칙 기반 자세 평가

`code/evaluation_functions.py`는 정규화 좌표(0~1) 위의 거리, 각도, 정렬로 자세 조건을 판정하는 함수 모음이다. `evaluate_condition`이 조건 문구를 받아 해당 함수로 분기하며, 분기된 조건 문구는 17개다. 대표 기준은 다음과 같다.

| 조건 | 판정 기준 |
|---|---|
| 수축 시 무릎과 팔꿈치가 충분히 가까움 | 팔꿈치-무릎 거리 < 0.1 (좌우 중 하나) |
| 무릎이 몸통 측면에서 올라오는지 | 골반-무릎-발목 각도 < 45도 |
| 앞다리 / 뒤다리 무릎 각도 90도 | 무릎 각도 85~95도 |
| 양손이 머리 뒤에 위치 | 손목과 귀의 x 차이 < 0.05 (양쪽) |
| 시선 정면 | 두 눈의 y 차이 < 0.02 |
| 상체의 과도한 숙임/젖힘 | 어깨-골반 y 차이가 ±0.1 이내 |

운동별 조건 목록은 별도 JSON(`동작조건.json`)에서 불러오고, 충족 개수로 단계를 정했다.

`code/deep2.ipynb`

```python
    for condition in conditions:
        condition_text = condition["condition"]
        result = evaluate_condition(landmarks, condition_text)
        evaluation_results[condition_text] = result

        if result:
            correct_count += 1
            score += 10  # 올바른 자세일 때 10점 추가
        else:
            combo_count = 0  # 틀리면 콤보 리셋

    # 평가 단계에 따라 추가 점수 부여
    if correct_count == len(conditions):
        score += 100  # 모든 조건이 맞으면 perfect
        stage = "Perfect"
    elif correct_count >= 3:
        score += 30  # 3개 이상 맞으면 good
        stage = "Good"
    else:
        score += 10  # 2개 이하 맞으면 bad
        stage = "Bad"
```

화면에는 운동명, 조건별 O/X 박스, `Status: Perfect/Good/Bad`를 Pillow로 그려 OpenCV 창에 표시했다.

## 4. 실험

### 4.1 실험 설정

| 항목 | 값 |
|---|---|
| 검증 방식 | 프레임 단위 무작위 80/20 홀드아웃, `random_state=42` |
| 배치 크기 | 32 |
| 옵티마이저 | Adam, lr 1e-3 |
| 손실 함수 | CrossEntropyLoss |
| 최대 epoch | 100 |
| 조기 종료 | val loss 기준 patience 20, delta 0 |
| 저장 | 종료 시점 모델 전체를 `torch.save(model, ...)`로 저장 (최적 epoch 복원 없음) |
| 환경 | CUDA 사용 가능 시 GPU, 아니면 CPU (노트북에 하드웨어 기록 없음) |

### 4.2 결과

발표 자료에 실린 결과는 `code/deep2.ipynb`의 학습 실행과 일치한다. 88 epoch에서 조기 종료됐고 최저 val loss는 68 epoch의 0.1417이었다.

| 클래스 | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 버피 테스트 | 0.95 | 0.95 | 0.95 | 824 |
| 스탠딩 니업 | 0.94 | 0.96 | 0.95 | 413 |
| 스탠딩 사이드 크런치 | 0.97 | 0.97 | 0.97 | 849 |
| 스텝 백워드 다이나믹 런지 | 0.82 | 0.85 | 0.83 | 169 |
| 스텝 포워드 다이나믹 런지 | 0.97 | 0.94 | 0.96 | 805 |
| Accuracy | | | 0.95 | 3,060 |
| Macro avg | 0.93 | 0.93 | 0.93 | 3,060 |
| Weighted avg | 0.95 | 0.95 | 0.95 | 3,060 |

가장 낮은 클래스는 검증 샘플이 가장 적은 스텝 백워드 다이나믹 런지(F1 0.83)였다. 같은 런지 계열인 스텝 포워드(805)보다 샘플 수가 5배가량 적다. 노트북에는 혼동 행렬, 클래스별 ROC 곡선, 클래스별 Precision/Recall/F1 막대 그래프가 이미지로 출력되어 있으나 AUC 값은 텍스트로 남아 있지 않아 생략한다.

### 4.3 비교 실험

| 변형 | 노트북 | 결과 | 처리 |
|---|---|---|---|
| TSM-Conv1D-BiLSTM, 좌표 100차원 (실행 1) | `deep.ipynb` | 71 epoch 조기 종료, 최저 val loss 0.1205 (51 epoch), Accuracy 0.95, macro F1 0.93, 스텝 백워드 런지 F1 0.84 | 동일 구조 재실행 |
| TSM-Conv1D-BiLSTM, 좌표 100차원 (실행 2) | `deep2.ipynb` | 88 epoch 조기 종료, 최저 val loss 0.1417 (68 epoch), Accuracy 0.95, macro F1 0.93 | 발표 결과로 사용 |
| 다중 view 이미지(5 view, 128x128 흑백) + 자세 조건 값 입력 Conv2D 분류기 | `deep3.ipynb` | 데이터 로더 동작만 확인 (`(1, 5, 48)` 좌표, `(1, 5, 1, 128, 128)` 이미지), 학습은 중단되어 결과 없음 | 미채택 |

실행 1과 2는 학습 코드와 분할 seed가 같고, 데이터 폴더 경로와 시각화 코드만 다르다. 두 실행 모두 Accuracy와 macro F1은 같고 클래스별 값만 0.01~0.04 차이가 났다. 이미지 기반 모델은 5개 view 이미지를 채널로 쌓은 Conv2D 2층 뒤에 조건 값 5개를 이어 붙이는 구조였지만, `64*64*64` 차원 FC 입력과 배치 크기 1 로딩으로 학습이 끝나지 않았다.

## 5. 한계와 향후 과제

- **시간축이 실제 프레임 순서가 아니다.** 학습 샘플은 프레임 1장이고, 모델의 "시간" 축은 한 프레임의 좌표 100개다. 채널 120개는 같은 벡터의 복제다. 따라서 TSM과 BiLSTM이 학습한 것은 프레임 간 동작이 아니라 좌표 배열 순서 위의 패턴이다. 연속 프레임을 (관절 좌표 x 시간) 형태로 묶은 시퀀스 샘플로 다시 구성해야 한다.
- **학습과 추론의 입력 형식이 다르다.** 학습은 AI Hub 2D view 평균 + 3D 좌표(100차원, 최댓값 정규화)를 채널 복제해 넣고, 추론은 MediaPipe 좌표 120개를 채널로, 2프레임을 시간축으로 넣는다(프레임별 [-1, 1] 정규화). 좌표계와 관절 순서도 다르다. 검증 정확도 0.95가 영상 추론 성능을 대표하지 않으며, 영상 단위 정량 평가가 없다.
- **검증셋 누수 가능성.** 프레임 단위 무작위 분할이라 같은 영상, 같은 사람의 인접 프레임이 학습과 검증에 함께 들어간다. 영상 또는 촬영자 단위 GroupKFold로 다시 측정해야 한다. 학습 데이터도 촬영일 폴더 하나로 제한되어 있다.
- **자세 평가 규칙의 검증 부재.** 임계값(0.05, 0.1, 45도, 85~95도)은 정규화 좌표 위의 고정값이고, JSON 라벨의 조건 충족 여부와 비교한 정확도 측정이 없다. `가슴의 충분한 이동`은 항상 True를 반환하는 미구현 상태이고, `시선 정면 유지`는 정의되지 않은 함수로 분기한다. 데모 노트북은 레포에 없는 `evaluation_functions3` 모듈과 `동작조건.json`을 불러온다. 라벨의 조건 정보를 정답으로 써서 임계값을 맞추거나 조건 분류 헤드를 학습하는 방향이 남아 있다.
- **저장 방식.** 모델 전체를 pickle로 저장해 같은 클래스 정의가 있어야 로드된다. `state_dict` 저장과 최적 epoch 가중치 복원이 필요하다.

## 참고

- 코드: https://github.com/tjehdgus/mediapipe_tsm_model
- 데이터: AI Hub 피트니스 자세 이미지 데이터
- 모델·도구: TSM (Temporal Shift Module for Efficient Video Understanding), Bidirectional LSTM, MediaPipe Pose (BlazePose), PyTorch, scikit-learn, OpenCV
