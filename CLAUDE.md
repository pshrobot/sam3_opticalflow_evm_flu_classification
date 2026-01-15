# Project Context - Chicken Detection/Segmentation

## 시스템 정보 - NVIDIA GB10 (DGX Spark)

**GB10은 Jetson이 아닙니다.** NVIDIA DGX Spark에 사용되는 Grace Blackwell Superchip입니다.

| 항목 | 사양 |
|------|------|
| **플랫폼** | NVIDIA GB10 (Grace Blackwell Superchip) |
| **OS** | DGX OS (Ubuntu 24.04 LTS 기반) |
| **CPU** | 20코어 ARM64 (10x Cortex-X925 + 10x Cortex-A725) |
| **GPU 아키텍처** | Blackwell SM_121 |
| **CUDA** | 13.0 |
| **드라이버** | 580.95.05 |
| **메모리** | 128GB 통합 메모리 (LPDDR5X) |
| **아키텍처** | aarch64 |

## 개발 환경
- **작업 디렉토리**: `/home/sanghyun/robotware/svi`
- **Python**: 3.12.3
- **PyTorch**: 2.11.0+cu130
- **CUDA**: 13.0

## 설치된 패키지
- ultralytics (YOLO)
- huggingface_hub
- sam3 (editable install from ./sam3)
- sam2
- einops, pycocotools, timm

## 현재 성능 (2026-01-04)

| 모델 | CV AUC | Test Accuracy |
|------|--------|---------------|
| 원본 Optical Flow | 92.73% | 84.62% |
| **원본 + EVM 결합** | **97.00%** | **90.67%** |

### 핵심 발견
- **병든 닭 특성**: 느린 움직임 + 일정한 방향
- **EVM 효과**: 미세 움직임 증폭으로 방향 패턴 강화

## 피처 엔지니어링 규칙

### 사용하는 피처 (806개)
- `mean_magnitude` histogram (100 bins) × 2 (원본 + EVM)
- `std_magnitude` histogram (100 bins) × 2
- `max_magnitude` histogram (100 bins) × 2
- `direction_uniformity` histogram (100 bins) × 2
- `activity_mean`, `activity_std`, `activity_max` × 2

### 사용하지 않는 피처 ⚠️
- **`area` 관련 피처**: `area_p00`~`area_p99`, `total_area`, `mean_area`
- **`num_ids`**: 감지된 개체 수
- **이유**: 카메라 위치/줌에 따라 변할 수 있어 일반화 성능 저하

## 완료된 작업

### 1. Optical Flow 기반 분류 파이프라인 (완료)
- SAM3 세그멘테이션 → ID별 Optical Flow → LightGBM 분류
- **최고 성능**: AUC 97.00%

### 2. EVM (Eulerian Video Magnification) 도입 (완료)
- 미세 움직임 증폭 → 원본과 결합하여 성능 향상
- **상세 보고서**: `output/evm_performance_report.md`

### 3. YOLO 닭 탐지 테스트 (성공)
- **모델**: `IceKhoffi/chicken-object-detection-yolov11s`
- **테스트 스크립트**: `test_chicken_yolo.py`
- **결과**: 이미지당 22-25마리 닭 탐지, 신뢰도 최대 82%
- **출력 디렉토리**: `output/chicken_detection/`

### 4. SAM3 테스트 (완료)
- **모델**: `facebook/sam3`
- **테스트 스크립트**: `test_sam3_chicken.py`
- **상태**: Hugging Face 접근 권한 승인 완료 ✓

## Hugging Face 인증
- 로그인 완료 (계정: `BettercallSaulGM`)
- SAM3 접근 권한 승인 완료

## 데이터
- 원본 비디오: `data/normal/*_raw.mp4`, `data/sick/*_raw.mp4`
- 프레임 이미지: `data/normal/frames/`, `data/sick/frames/`
- SAM3 마스크: `output/sam3_masks/`
- EVM 프레임: `output/evm_frames/`

## 주요 스크립트
| 스크립트 | 설명 |
|---------|------|
| `apply_evm.py` | EVM 적용하여 증폭 프레임 생성 |
| `extract_optical_flow_per_id.py` | ID별 Optical Flow 통계 추출 |
| `extract_optical_flow_evm.py` | EVM 프레임에서 Optical Flow 추출 |
| `create_lightgbm_features.py` | 분위수 히스토그램 피처 생성 |
| `train_lightgbm_id_features.py` | LightGBM 모델 학습 |

## 참고 사항
- decord 패키지는 aarch64에서 사용 불가 → mock 모듈로 대체 (`./venv/lib/python3.12/site-packages/decord/__init__.py`)
- SAM3 bpe_path: `/home/sanghyun/robotware/svi/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz`

## DGX Spark / GB10 Triton 관련

### Triton Inference Server 설치
DGX Spark는 Triton Inference Server가 미리 설치되어 출시됨:
```bash
tritonserver --version  # 설치 확인
```

### NGC 컨테이너 사용 시 주의
- **arm64sbsa 태그 필수**: 표준 x86_64 이미지는 실행 불가
```bash
docker pull nvcr.io/nvidia/tritonserver:24.08-py3-arm64sbsa
```

### SM_121 관련 알려진 문제
- NIM LLM 컨테이너에서 SM_121 관련 Triton/vLLM 크래시 문제 보고됨
- 해결책: `VLLM_MXFP4_USE_MARLIN=1` 환경변수로 Triton 대신 Marlin 사용
- 일부 케이스에서 PyTorch/Triton을 SM_121 + CUDA 13.0용으로 직접 컴파일 필요

### 참고 자료
- [DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- NGC 컨테이너: ARM64 SBSA 버전 선택 필요
