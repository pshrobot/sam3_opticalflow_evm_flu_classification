# EVM 도입 성능 비교 보고서

## 요약

Eulerian Video Magnification (EVM) 도입으로 **AUC 4.27%p, Accuracy 6.05%p 향상** 달성.
(영역/개체수 피처 제외 기준)

| 모델 | CV AUC | Test AUC | Test Accuracy |
|------|--------|----------|---------------|
| 원본 (ID별 합집합) | 92.73% | - | 84.62% |
| EVM 단독 | 90.47% | 89.16% | 80.45% |
| 결합 (전체 피처) | 98.41% | 98.23% | 93.34% |
| **결합 (영역 제외)** | **97.00%** | **96.22%** | **90.67%** |

> **Note**: 영역(area) 및 개체수(num_ids) 피처는 카메라 위치/줌에 따라 변할 수 있어 일반화를 위해 제외함.

---

## 1. 실험 설정

### 1.1 데이터셋
| 항목 | Normal | Sick |
|------|--------|------|
| 비디오 길이 | ~6,325초 | ~6,325초 |
| 프레임 수 (1fps) | 5,066 | 5,065 |
| ID 추출 수 | ~1.68M | ~1.42M |

### 1.2 EVM 파라미터
```
freq_min: 0.5 Hz
freq_max: 2.0 Hz
amplification: 20
pyramid_levels: 4
처리 시간: ~12분/비디오
```

### 1.3 피처 구성 (영역 제외 버전)

| 피처 타입 | 원본 | EVM | 결합 |
|----------|------|-----|------|
| magnitude histogram (100bins) | 100 | 100 | 200 |
| std_magnitude histogram | 100 | 100 | 200 |
| max_magnitude histogram | 100 | 100 | 200 |
| direction_uniformity histogram | 100 | 100 | 200 |
| activity 통계 (mean, std, max) | 3 | 3 | 6 |
| **총 피처** | **403** | **403** | **806** |

**제외된 피처:**
- `area_p00` ~ `area_p99` (100개 × 2)
- `num_ids`, `total_area`, `mean_area` (3개 × 2)
- 총 206개 피처 제외

---

## 2. 모델별 성능 비교

### 2.1 Cross Validation (5-Fold)

| 지표 | 원본 | EVM 단독 | 결합 (전체) | 결합 (영역 제외) |
|------|------|----------|-------------|------------------|
| AUC | 92.73% | 90.47% | 98.41% | **97.00%** |
| Accuracy | 84.62% | 82.96% | 93.78% | **91.04%** |
| Precision | - | 83.03% | 93.83% | **91.12%** |
| Recall | - | 82.84% | 93.73% | **90.94%** |
| F1 Score | - | 82.93% | 93.78% | **91.03%** |

### 2.2 Test Set 성능 (영역 제외 모델)

| 지표 | 값 |
|------|-----|
| AUC | 96.22% |
| Accuracy | 90.67% |
| Precision | 90.23% |
| Recall | 91.21% |
| F1 Score | 90.72% |

### 2.3 Confusion Matrix (영역 제외 모델)
```
                Predicted
                Normal    Sick
Actual Normal     913      100
Actual Sick        89      924

Sensitivity (Recall): 91.21%
Specificity: 90.13%
False Negative Rate: 8.79%
```

---

## 3. Feature Importance 분석 (영역 제외 모델)

### 3.1 Top 20 중요 피처
| 순위 | 피처명 | 중요도 | 출처 | 의미 |
|------|--------|--------|------|------|
| 1 | evm_dir_uni_p99 | 7,920 | EVM | 높은 방향 일관성 비율 |
| 2 | mag_mean_p00 | 5,597 | 원본 | 낮은 움직임 비율 |
| 3 | activity_mean | 4,278 | 원본 | 평균 활동량 |
| 4 | dir_uni_p99 | 3,187 | 원본 | 높은 방향 일관성 비율 |
| 5 | mag_mean_p01 | 2,490 | 원본 | 낮은 움직임 비율 |
| 6 | evm_activity_mean | 2,098 | EVM | EVM 평균 활동량 |
| 7 | evm_mag_mean_p00 | 1,905 | EVM | EVM 낮은 움직임 비율 |
| 8 | mag_mean_p02 | 1,332 | 원본 | 낮은 움직임 비율 |
| 9 | dir_uni_p91 | 810 | 원본 | 방향 일관성 |
| 10 | dir_uni_p88 | 759 | 원본 | 방향 일관성 |

### 3.2 핵심 발견

**병든 닭의 특성:**
1. **느린 움직임**: `mag_mean_p00~p02` (낮은 magnitude 구간 비율 높음)
2. **일정한 방향**: `dir_uni_p99` (방향 일관성 높음)
3. **낮은 활동량**: `activity_mean` 낮음

**EVM 기여:**
- `evm_dir_uni_p99`: 가장 중요한 피처 (EVM에서 방향 패턴이 더 명확)
- EVM은 미세한 움직임을 증폭하여 방향 일관성 특성을 강화

---

## 4. 결론 및 시사점

### 4.1 EVM 효과
| 비교 | AUC 변화 |
|------|----------|
| 원본 → 결합 (영역 제외) | +4.27%p |
| 원본 → 결합 (전체) | +5.68%p |

### 4.2 영역 피처 제외 영향
- AUC 감소: -1.41%p (98.41% → 97.00%)
- **일반화 관점에서 영역 제외 권장** (카메라 설정 독립적)

### 4.3 핵심 분류 요소
1. **움직임 크기** (magnitude): 병든 닭 = 느린 움직임
2. **움직임 방향** (direction_uniformity): 병든 닭 = 일정한 방향
3. **EVM 증폭 효과**: 미세 움직임 패턴 강화

### 4.4 향후 개선 방향
1. **EVM 파라미터 최적화**: freq 범위, amplification 튜닝
2. **시계열 모델**: LSTM/Transformer로 연속 패턴 학습
3. **실시간 추론**: 경량화 및 최적화

---

## 5. 파일 위치

```
output/
├── evm_frames/                        # EVM 증폭 프레임
├── optical_flow_evm/                  # EVM Optical Flow 통계
├── features_evm/                      # EVM 피처
├── features_combined/                 # 결합 피처
│   ├── combined_features.csv          # 전체 피처
│   └── combined_features_no_area.csv  # 영역 제외 피처
├── models_combined/                   # 결합 모델 (전체 피처)
└── models_combined_no_area/           # 결합 모델 (영역 제외, 권장)
```

---

## 6. 최종 권장 모델

**모델**: `models_combined_no_area/lightgbm_model_*.txt`

| 지표 | 값 |
|------|-----|
| CV AUC | 97.00% |
| Test AUC | 96.22% |
| Test Accuracy | 90.67% |
| 피처 수 | 806 |

---

*생성일: 2026-01-04*
*업데이트: 영역/개체수 피처 제외 버전 추가*
