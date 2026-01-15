# SVI - Static camera Visual Intelligence

양계장 CCTV 영상 기반 닭 건강 상태 분류 시스템

## 프로젝트 개요

Optical Flow + EVM (Eulerian Video Magnification) 분석을 통해 정상(normal)과 병든(sick) 닭을 분류하는 머신러닝 파이프라인입니다.

## 현재 성능 (2026-01-04)

| 모델 | CV AUC | Test Accuracy |
|------|--------|---------------|
| 원본 Optical Flow | 92.73% | 84.62% |
| **원본 + EVM 결합** | **97.00%** | **90.67%** |

### 핵심 발견
- **병든 닭 특성**: 느린 움직임 + 일정한 방향성
- **EVM 효과**: 미세 움직임 증폭으로 방향 패턴 강화 → **+4.27%p AUC 향상**

## 환경

| 항목 | 사양 |
|------|------|
| **플랫폼** | NVIDIA GB10 (DGX Spark) |
| **OS** | DGX OS (Ubuntu 24.04 LTS) |
| **Python** | 3.12.3 |
| **PyTorch** | 2.11.0+cu130 |
| **CUDA** | 13.0 |
| **아키텍처** | aarch64 |

## 설치

```bash
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision opencv-python numpy pandas lightgbm tqdm scipy
uv pip install -e ./sam3  # SAM3 설치
```

## 파이프라인 개요

```
원본 비디오 (*.mp4)
       ↓
[1] 프레임 추출 (1fps)
       ↓
[2] SAM3 세그멘테이션 → 닭 마스크 (ID 추적)
       ↓
   ┌───┴───┐
   ↓       ↓
[3] EVM 적용 (미세 움직임 증폭)
   ↓       ↓
[4-A] Optical Flow (원본)    [4-B] Optical Flow (EVM)
   ↓       ↓
   └───┬───┘
       ↓
[5] 피처 생성 (806개 피처)
       ↓
[6] LightGBM 학습/예측
```

## 실행 순서 (Raw Data → 학습)

### Step 1. 프레임 추출

비디오에서 1fps로 프레임 추출:

```bash
# Normal 데이터
python extract_video_frames.py \
  --video data/normal/D01_raw.mp4 \
  --output_dir data/normal/frames \
  --fps 1

# Sick 데이터
python extract_video_frames.py \
  --video data/sick/D01_raw.mp4 \
  --output_dir data/sick/frames \
  --fps 1
```

### Step 2. SAM3 마스크 추출

프레임에서 닭 세그멘테이션 및 ID 추적:

```bash
python extract_sam3_masks.py \
  --data_dir data \
  --output_dir output/sam3_masks \
  --text_prompt "chicken" \
  --datasets normal sick
```

### Step 3. EVM 적용

미세 움직임 증폭 프레임 생성:

```bash
# Normal 데이터
python apply_evm.py \
  --video data/normal/D01_raw.mp4 \
  --output_dir output/evm_frames/normal \
  --freq_min 0.5 \
  --freq_max 2.0 \
  --amplification 20

# Sick 데이터
python apply_evm.py \
  --video data/sick/D01_raw.mp4 \
  --output_dir output/evm_frames/sick \
  --freq_min 0.5 \
  --freq_max 2.0 \
  --amplification 20
```

### Step 4-A. Optical Flow 추출 (원본)

원본 프레임에서 ID별 Optical Flow 통계 추출:

```bash
python extract_optical_flow_per_id.py \
  --data_dir data \
  --masks_dir output/sam3_masks \
  --output_dir output/optical_flow \
  --datasets normal sick
```

### Step 4-B. Optical Flow 추출 (EVM)

EVM 프레임에서 Optical Flow 통계 추출:

```bash
python extract_optical_flow_evm.py \
  --evm_dir output/evm_frames \
  --masks_dir output/sam3_masks \
  --output_dir output/optical_flow_evm \
  --datasets normal sick
```

### Step 5. 피처 생성

분위수 히스토그램 피처 생성:

```bash
python create_lightgbm_features.py \
  --input_dir output/optical_flow \
  --output_dir output/features
```

### Step 6. 모델 학습

LightGBM 분류기 학습:

```bash
python train_lightgbm_id_features.py \
  --features_path output/features/features.csv \
  --output_dir output/models \
  --cv_folds 5
```

> **옵션**: `--include_area_count` 플래그로 area/count 피처 포함 가능 (권장하지 않음)

## 실행 순서 (예측)

학습된 모델로 새 비디오 예측:

```bash
# Step 1~4: 위와 동일하게 수행

# Step 5: 예측
python predict_lightgbm.py \
  --features_path output/new_video/features/features.csv \
  --model_path output/models/lightgbm_model.txt \
  --output_dir output/new_video/predictions
```

## 주요 스크립트

| 스크립트 | 설명 |
|---------|------|
| `extract_video_frames.py` | 비디오에서 프레임 추출 (1fps) |
| `extract_sam3_masks.py` | SAM3로 닭 세그멘테이션 및 ID 추적 |
| `apply_evm.py` | Eulerian Video Magnification 적용 |
| `extract_optical_flow_per_id.py` | 원본 프레임에서 ID별 Optical Flow 추출 |
| `extract_optical_flow_evm.py` | EVM 프레임에서 Optical Flow 추출 |
| `create_lightgbm_features.py` | 분위수 히스토그램 피처 생성 |
| `train_lightgbm_id_features.py` | LightGBM 모델 학습 |
| `predict_lightgbm.py` | 학습된 모델로 예측 수행 |

## 피처 구성 (806개 - 원본+EVM 결합 시)

### 히스토그램 피처 (800개)
- `mean_magnitude` × 100 bins × 2 (원본 + EVM)
- `std_magnitude` × 100 bins × 2
- `max_magnitude` × 100 bins × 2
- `direction_uniformity` × 100 bins × 2

### 집계 피처 (6개)
- `activity_mean`, `activity_std`, `activity_max` × 2 (원본 + EVM)

### 사용하지 않는 피처
- `area` 관련: 카메라 위치/줌에 따라 변동
- `num_ids`: 일반화 성능 저하 원인

## 디렉토리 구조

```
svi/
├── data/                       # 원본 데이터 (gitignore)
│   ├── normal/                 # 정상 닭 비디오
│   │   ├── D01_raw.mp4
│   │   └── frames/             # 추출된 프레임
│   └── sick/                   # 병든 닭 비디오
│       ├── D01_raw.mp4
│       └── frames/
├── output/                     # 생성된 결과물 (gitignore)
│   ├── sam3_masks/             # SAM3 세그멘테이션 마스크
│   │   ├── normal/mask/
│   │   └── sick/mask/
│   ├── evm_frames/             # EVM 적용된 프레임
│   │   ├── normal/
│   │   └── sick/
│   ├── optical_flow/           # 원본 Optical Flow 통계
│   │   ├── normal/frames/
│   │   └── sick/frames/
│   ├── optical_flow_evm/       # EVM Optical Flow 통계
│   ├── features/               # LightGBM 피처
│   │   └── features.csv
│   └── models/                 # 학습된 모델
│       ├── lightgbm_model.txt
│       └── bin_edges.json
├── sam3/                       # SAM3 패키지
├── docs/                       # 문서
│   ├── evm_performance_report.md
│   └── optical_flow_classification_report.md
├── extract_video_frames.py
├── extract_sam3_masks.py
├── apply_evm.py
├── extract_optical_flow_per_id.py
├── extract_optical_flow_evm.py
├── create_lightgbm_features.py
├── train_lightgbm_id_features.py
├── predict_lightgbm.py
├── CLAUDE.md                   # AI 개발 컨텍스트
└── README.md
```

## 참고 자료

- [Eulerian Video Magnification](https://people.csail.mit.edu/mrub/evm/)
- [SAM3 - Segment Anything Model 3](https://github.com/facebookresearch/sam3)
- [LightGBM](https://lightgbm.readthedocs.io/)
