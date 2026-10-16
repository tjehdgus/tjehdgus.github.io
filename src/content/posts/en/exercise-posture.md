---
title: "Exercise Posture Classification and Evaluation: A TSM-BiLSTM Classifier on AI Hub Joint Coordinates with Rule-Based MediaPipe Scoring"
description: "A TSM-Conv1D-BiLSTM model trained on 2D and 3D joint coordinates from the AI Hub fitness posture dataset classifies five exercises, and rule-based posture checks on MediaPipe Pose landmarks label each video frame Perfect, Good or Bad. On a 3,060-frame validation split the classifier reached 0.95 accuracy and 0.93 macro F1."
date: 2026-03-05
lang: en
key: exercise-posture
tags: [Pose Estimation, MediaPipe, TSM, BiLSTM, PyTorch, Action Recognition]
---

## Abstract

This 2024 team project from a training program built a system that identifies the exercise in a workout video and evaluates the user's posture. Training data came from the joint-coordinate JSON labels of the AI Hub fitness posture dataset; each frame was turned into a 100-dimensional vector by averaging 2D coordinates across camera views and appending the 3D coordinates. The classifier stacks a temporal shift layer, Conv1D, BatchNorm, a bidirectional LSTM and two fully connected layers, and predicts one of five exercises. At inference time MediaPipe Pose extracts landmarks from the video, the classifier fixes the exercise label, and rule functions check exercise-specific posture conditions to display a Perfect/Good/Bad stage. On a random 20% frame-level validation split of 3,060 frames the model reached 0.95 accuracy, 0.93 macro F1 and 0.95 weighted F1. I uploaded and organized the repository, but its commit history does not separate individual contributions, so the work is described at team level.

## 1. Problem

- Input (training): one frame's joint-coordinate vector (x, y for 20 joints in 2D, plus x, y, z for 20 joints in 3D)
- Input (inference): a workout video (mp4), with per-frame landmarks from MediaPipe Pose
- Output 1: one of five exercises (burpee test, standing knee-up, standing side crunch, step-backward dynamic lunge, step-forward dynamic lunge)
- Output 2: pass/fail for each posture condition of that exercise, and a stage (Perfect / Good / Bad)
- Metrics: per-class precision, recall and F1, overall accuracy, macro and weighted averages, and per-class ROC curves for classification. Posture evaluation was checked only on demo videos, with no quantitative metric.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
macro avg = unweighted mean over classes, weighted avg = mean weighted by class support
```

The main constraint was compute. The presentation notes that the raw dataset exceeded 1 TB and could not be trained at once because of memory limits. The training notebooks use only the labeled JSON files from a single capture-day folder (`Day13_201013_F`).

## 2. Data

| Item | Value |
|---|---|
| Source | AI Hub fitness posture image dataset (labeling JSON) |
| Scope used | 2D JSON files from one capture-day folder and the matching `-3d.json` files |
| Sample unit | one frame = one sample |
| Classes | 5 exercises (`type_info.exercise`) |
| Features | 20 joints x 2D (x, y) = 40, plus 20 joints x 3D (x, y, z) = 60, total 100 |
| Split | `train_test_split(test_size=0.2, random_state=42)`, not stratified |
| Validation set | 3,060 frames |

Validation class counts, from the support column of the classification report in `code/deep2.ipynb`:

| Class | Validation frames |
|---|---|
| Burpee test | 824 |
| Standing knee-up | 413 |
| Standing side crunch | 849 |
| Step-backward dynamic lunge | 169 |
| Step-forward dynamic lunge | 805 |

Besides the exercise name, each JSON file holds three to five posture conditions and whether each is satisfied (per the presentation). The classifier used only the exercise name as its label.

### 2.1 Merging 2D and 3D coordinates

The 2D labels contain several camera views per frame, each with its own joint coordinates, while the 3D labels have a single (x, y, z) per joint and no views. To merge them into one vector, the 2D coordinates were averaged over views and the 3D coordinates for the same frame index were appended. The 20 joints are nose, left/right eye, ear, shoulder, elbow, wrist, hip, knee and ankle, plus neck, back and waist.

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

Frames without a 3D file were zero-padded to the longest vector length, and all values were divided by the dataset-wide maximum.

## 3. Method

### 3.1 Pipeline

```text
[Training]
AI Hub labeling JSON (multi-view 2D + 3D)
  └─ per-frame coordinate merge (2D view mean 40 + 3D 60 = 100 dims)
       └─ zero padding + max normalization
            └─ random 80/20 split (seed 42)
                 └─ repeat 120 times along the channel axis → (B, 120, 100)
                      └─ TSM-Conv1D-BiLSTM classifier (5 classes)
                           └─ EarlyStopping (patience 20) → save full model (.pth)

[Inference]
workout video (mp4)
  ├─ pass 1: MediaPipe Pose (model_complexity=2)
  │    ├─ landmark extraction + derived neck/pelvis/spine → padded to 120 values
  │    ├─ exponential smoothing (alpha 0.8) + per-frame min-max to [-1, 1]
  │    ├─ two consecutive frames fed to the classifier
  │    └─ mode of all predictions = exercise label for the video
  └─ pass 2: per-frame posture evaluation
       ├─ look up the posture conditions for that exercise
       ├─ rule function per condition → pass/fail
       └─ number of passed conditions → Perfect / Good / Bad overlay
```

- Coordinate merge: brings multi-view 2D and single-view 3D labels to one fixed-length vector.
- Classifier: a shift layer and Conv1D for local patterns, a BiLSTM for order information.
- Exercise label: per-frame predictions fluctuate, so a single label per video was fixed by majority vote.
- Posture evaluation: rule functions on landmark distances, angles and alignment, not a learned model.

### 3.2 Classifier

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

The tensor shapes below follow the code. The implemented TSM layer does not shift a fraction of channels along time as in the original paper; it concatenates two slices offset by one step along the time axis, so the length grows from `T` to `2T-2`.

| Layer | Setting | Output shape (B=32) |
|---|---|---|
| Input | 100-dim vector repeated 120 times on the channel axis | (32, 120, 100) |
| TSMLayer | shift_size=1, concatenation of two slices | (32, 120, 198) |
| Conv1d | 120 → 64, kernel 3, padding 1 | (32, 64, 198) |
| MaxPool1d + BatchNorm1d | pool 2 | (32, 64, 99) |
| BiLSTM | hidden 128, bidirectional, last step | (32, 256) |
| Dropout + FC + ReLU | p=0.2, 256 → 64 | (32, 64) |
| FC | 64 → 5 | (32, 5) |

Computed from the layer definitions, the model has 238,661 trainable parameters (BiLSTM 198,656, Conv1d 23,104, FC 16,773, BatchNorm 128). Tensor sizes in the saved `model/tsm_model_full.pth` (Conv1d weight 64x120x3, final FC 5x64) match this definition. The layer table in the presentation lists sequence lengths of 100 and 50 after the convolution, whereas the code produces 198 and 99.

### 3.3 MediaPipe landmarks and derived joints

MediaPipe Pose ran with `static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5`. Neck, pelvis and spine exist in the AI Hub labels but not in MediaPipe, so they were derived as landmark midpoints. The four eye and ear entries were mapped to `None` and filled with (0, 0, 0).

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

The 21 mapped joints and three derived joints give 72 values as (x, y, z), zero-padded to 120. Each frame was then smoothed against the previous one as `0.8 * prev + 0.2 * new` and min-max normalized to [-1, 1] within the frame. Two consecutive frames were fed to the classifier as a `(1, 120, 2)` tensor, and the most frequent prediction over the video became the exercise label.

### 3.4 Rule-based posture evaluation

`code/evaluation_functions.py` collects functions that judge posture conditions from distances, angles and alignment on normalized (0 to 1) landmark coordinates. `evaluate_condition` dispatches on the condition text and covers 17 condition strings. Representative rules:

| Condition | Rule |
|---|---|
| Knee and elbow close at contraction | elbow-knee distance < 0.1 (either side) |
| Knee raised from the side of the torso | hip-knee-ankle angle < 45 degrees |
| Front / back knee at 90 degrees | knee angle within 85 to 95 degrees |
| Both hands behind the head | wrist-ear x gap < 0.05 (both sides) |
| Eyes facing forward | eye y gap < 0.02 |
| No excessive forward/backward lean | shoulder-hip y gap within ±0.1 |

The condition list per exercise was loaded from a separate JSON file (`동작조건.json`), and the stage was set by the number of satisfied conditions.

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

The exercise name, a pass/fail box per condition and `Status: Perfect/Good/Bad` were drawn with Pillow and shown in an OpenCV window.

## 4. Experiments

### 4.1 Setup

| Item | Value |
|---|---|
| Validation | random frame-level 80/20 hold-out, `random_state=42` |
| Batch size | 32 |
| Optimizer | Adam, lr 1e-3 |
| Loss | CrossEntropyLoss |
| Max epochs | 100 |
| Early stopping | on validation loss, patience 20, delta 0 |
| Checkpoint | full model saved with `torch.save(model, ...)` at stop time (best epoch not restored) |
| Hardware | GPU if CUDA is available, otherwise CPU (not recorded in the notebooks) |

### 4.2 Results

The results in the presentation match the training run in `code/deep2.ipynb`. Training stopped early at epoch 88; the lowest validation loss was 0.1417 at epoch 68.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Burpee test | 0.95 | 0.95 | 0.95 | 824 |
| Standing knee-up | 0.94 | 0.96 | 0.95 | 413 |
| Standing side crunch | 0.97 | 0.97 | 0.97 | 849 |
| Step-backward dynamic lunge | 0.82 | 0.85 | 0.83 | 169 |
| Step-forward dynamic lunge | 0.97 | 0.94 | 0.96 | 805 |
| Accuracy | | | 0.95 | 3,060 |
| Macro avg | 0.93 | 0.93 | 0.93 | 3,060 |
| Weighted avg | 0.95 | 0.95 | 0.95 | 3,060 |

The weakest class was the step-backward dynamic lunge (F1 0.83), which also had the fewest validation samples, about five times fewer than the related step-forward lunge (805). The notebook renders a confusion matrix, per-class ROC curves and a per-class precision/recall/F1 bar chart as images, but the AUC values are not stored as text, so they are not reported here.

### 4.3 Ablation

| Variant | Notebook | Result | Decision |
|---|---|---|---|
| TSM-Conv1D-BiLSTM, 100-dim coordinates (run 1) | `deep.ipynb` | stopped at epoch 71, lowest val loss 0.1205 (epoch 51), accuracy 0.95, macro F1 0.93, step-backward lunge F1 0.84 | rerun of the same architecture |
| TSM-Conv1D-BiLSTM, 100-dim coordinates (run 2) | `deep2.ipynb` | stopped at epoch 88, lowest val loss 0.1417 (epoch 68), accuracy 0.95, macro F1 0.93 | reported result |
| Conv2D classifier on multi-view images (5 views, 128x128 grayscale) plus posture-condition values | `deep3.ipynb` | only the data loader was verified (`(1, 5, 48)` coordinates, `(1, 5, 1, 128, 128)` images); training was interrupted with no result | not adopted |

Runs 1 and 2 share the training code and split seed and differ only in the data folder path and plotting code. Both reached the same accuracy and macro F1, with per-class values differing by 0.01 to 0.04. The image model stacked five view images as channels, applied two Conv2D layers and concatenated five condition values before the FC layers, but with a `64*64*64`-dimensional FC input and batch-size-1 loading its training never finished.

## 5. Limitations and future work

- **The time axis is not frame order.** Each training sample is a single frame, and the model's "time" axis is the 100 coordinates of that frame; the 120 channels are copies of the same vector. The TSM layer and BiLSTM therefore learn patterns over coordinate order, not motion across frames. Samples should be rebuilt as sequences of consecutive frames (joint coordinates x time).
- **Training and inference inputs differ.** Training feeds the AI Hub 2D view mean plus 3D coordinates (100 dims, max-normalized) replicated across channels; inference feeds 120 MediaPipe values as channels and two frames as time, with per-frame [-1, 1] normalization. Coordinate systems and joint order also differ. The 0.95 validation accuracy does not represent video inference performance, and there is no video-level quantitative evaluation.
- **Likely validation leakage.** The split is random at frame level, so adjacent frames of the same video and subject fall into both train and validation. A group split by video or subject is needed. Training data is also limited to one capture-day folder.
- **Posture rules are unvalidated.** Thresholds (0.05, 0.1, 45 degrees, 85 to 95 degrees) are fixed values on normalized coordinates and were never compared against the condition labels in the JSON. `가슴의 충분한 이동` (sufficient chest movement) always returns True, and `시선 정면 유지` dispatches to an undefined function. The demo notebook imports an `evaluation_functions3` module and `동작조건.json` that are not in the repository. Using the label conditions as ground truth to tune thresholds, or training a condition-classification head, remains open.
- **Checkpoint format.** The full model is pickled, so loading requires the same class definitions. Saving a `state_dict` and restoring the best-epoch weights would be needed.

## References

- Code: https://github.com/tjehdgus/mediapipe_tsm_model
- Data: AI Hub fitness posture image dataset
- Models and tools: TSM (Temporal Shift Module for Efficient Video Understanding), Bidirectional LSTM, MediaPipe Pose (BlazePose), PyTorch, scikit-learn, OpenCV
