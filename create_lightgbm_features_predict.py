#!/usr/bin/env python3
"""
예측용 LightGBM 피처 생성 스크립트
- 원본 + EVM optical flow를 결합
- 학습 시 저장된 bin_edges 사용 (새로 계산하지 않음)
- area 관련 피처 제외 (806개 피처)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any
import argparse


# 히스토그램 피처 (area 제외)
HISTOGRAM_FEATURES = [
    'mean_magnitude',
    'std_magnitude',
    'max_magnitude',
    'direction_uniformity'
]

FEATURE_PREFIXES = {
    'mean_magnitude': 'mag_mean',
    'std_magnitude': 'mag_std',
    'max_magnitude': 'mag_max',
    'direction_uniformity': 'dir_uni'
}

N_BINS = 100


def load_bin_edges(bin_edges_path: Path) -> Dict[str, np.ndarray]:
    """학습 시 저장된 bin edges 로드"""
    print(f"Loading bin edges from: {bin_edges_path}")
    with open(bin_edges_path, 'r') as f:
        data = json.load(f)

    bin_edges = {}
    for feat in HISTOGRAM_FEATURES:
        if feat in data:
            bin_edges[feat] = np.array(data[feat])
            print(f"  {feat}: {len(bin_edges[feat])} edges")

    return bin_edges


def load_frame_statistics(frames_dir: Path) -> Dict[int, Dict]:
    """frames 디렉토리에서 개별 JSON 파일들을 로드"""
    if not frames_dir.exists():
        print(f"  Directory not found: {frames_dir}")
        return {}

    frame_files = sorted(frames_dir.glob('frame_*.json'))
    data = {}

    for frame_file in tqdm(frame_files, desc="  Loading"):
        with open(frame_file, 'r') as f:
            frame_data = json.load(f)
            frame_idx = frame_data.get('frame_idx', int(frame_file.stem.split('_')[1]))
            data[frame_idx] = frame_data

    print(f"  Loaded {len(data)} frames")
    return data


def create_percentile_histogram(values: List[float], bin_edges: np.ndarray) -> np.ndarray:
    """분위수 구간별 비율 계산"""
    if len(values) == 0:
        return np.zeros(N_BINS)

    values = np.array(values)
    bin_indices = np.digitize(values, bin_edges)
    hist = np.bincount(bin_indices, minlength=N_BINS)

    # 정확히 N_BINS 크기로 맞추기
    if len(hist) > N_BINS:
        hist = hist[:N_BINS]

    return hist / len(values)


def create_frame_features(
    frame_data: Dict,
    bin_edges: Dict[str, np.ndarray],
    prefix: str = ""
) -> Dict[str, float]:
    """프레임 하나에 대한 피처 생성"""
    features = {}

    # union_ids와 standalone_ids 합쳐서 처리
    ids = frame_data.get('union_ids', []) + frame_data.get('standalone_ids', [])

    # 히스토그램 피처 생성 (area 제외)
    for feat in HISTOGRAM_FEATURES:
        values = [id_stat[feat] for id_stat in ids if feat in id_stat]
        hist = create_percentile_histogram(values, bin_edges[feat])

        # prefix가 있으면 앞에 붙임 (예: evm_mag_mean)
        feat_prefix = f"{prefix}{FEATURE_PREFIXES[feat]}" if prefix else FEATURE_PREFIXES[feat]
        for i, ratio in enumerate(hist):
            features[f'{feat_prefix}_p{i:02d}'] = ratio

    # 추가 집계 피처 (area 제외)
    # 학습 데이터와 동일한 이름 사용: activity_mean, activity_mean_evm
    suffix = "_evm" if prefix == "evm_" else ""
    if ids:
        magnitudes = [id_stat['mean_magnitude'] for id_stat in ids if 'mean_magnitude' in id_stat]
        if magnitudes:
            features[f'activity_mean{suffix}'] = np.mean(magnitudes)
            features[f'activity_std{suffix}'] = np.std(magnitudes)
            features[f'activity_max{suffix}'] = np.max(magnitudes)
        else:
            features[f'activity_mean{suffix}'] = 0.0
            features[f'activity_std{suffix}'] = 0.0
            features[f'activity_max{suffix}'] = 0.0
    else:
        features[f'activity_mean{suffix}'] = 0.0
        features[f'activity_std{suffix}'] = 0.0
        features[f'activity_max{suffix}'] = 0.0

    return features


def create_combined_features(
    optical_flow_dir: Path,
    optical_flow_evm_dir: Path,
    output_dir: Path,
    bin_edges_path: Path,
) -> pd.DataFrame:
    """
    원본 + EVM optical flow를 결합한 피처 생성

    Args:
        optical_flow_dir: 원본 optical flow frames 디렉토리
        optical_flow_evm_dir: EVM optical flow frames 디렉토리
        output_dir: 출력 디렉토리
        bin_edges_path: 학습 시 저장된 bin_edges.json 경로

    Returns:
        피처 DataFrame
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Combined Feature Generation for Prediction")
    print("=" * 60)
    print(f"Original OF: {optical_flow_dir}")
    print(f"EVM OF: {optical_flow_evm_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # bin_edges 로드
    bin_edges = load_bin_edges(bin_edges_path)

    # 원본 optical flow 데이터 로드
    print("\nLoading original optical flow data...")
    orig_data = load_frame_statistics(optical_flow_dir)

    # EVM optical flow 데이터 로드
    print("\nLoading EVM optical flow data...")
    evm_data = load_frame_statistics(optical_flow_evm_dir)

    # 공통 프레임 인덱스 찾기
    common_frames = sorted(set(orig_data.keys()) & set(evm_data.keys()))
    print(f"\nCommon frames: {len(common_frames)}")

    if not common_frames:
        # 공통 프레임이 없으면 원본만 사용
        print("No common frames found. Using original only.")
        common_frames = sorted(orig_data.keys())
        use_evm = False
    else:
        use_evm = True

    # 피처 생성
    all_features = []

    print("\nCreating features...")
    for frame_idx in tqdm(common_frames, desc="  Processing"):
        features = {'frame_idx': frame_idx}

        # 원본 피처
        if frame_idx in orig_data:
            orig_features = create_frame_features(orig_data[frame_idx], bin_edges, prefix="")
            features.update(orig_features)

        # EVM 피처
        if use_evm and frame_idx in evm_data:
            evm_features = create_frame_features(evm_data[frame_idx], bin_edges, prefix="evm_")
            features.update(evm_features)

        all_features.append(features)

    # DataFrame 생성
    df = pd.DataFrame(all_features)

    # 누락된 컬럼 0으로 채우기
    df = df.fillna(0)

    # 피처 순서: 학습 데이터와 동일한 순서 유지
    # 원본: mag_mean, mag_std, mag_max, dir_uni (각 100 bins)
    # EVM: evm_mag_mean, evm_mag_std, evm_mag_max, evm_dir_uni (각 100 bins)
    # 집계: activity_mean, activity_std, activity_max, activity_mean_evm, activity_std_evm, activity_max_evm
    expected_order = []

    # 원본 히스토그램 피처
    for feat_prefix in ['mag_mean', 'mag_std', 'mag_max', 'dir_uni']:
        for i in range(N_BINS):
            expected_order.append(f'{feat_prefix}_p{i:02d}')

    # 원본 집계 피처
    expected_order.extend(['activity_mean', 'activity_std', 'activity_max'])

    # EVM 히스토그램 피처
    for feat_prefix in ['evm_mag_mean', 'evm_mag_std', 'evm_mag_max', 'evm_dir_uni']:
        for i in range(N_BINS):
            expected_order.append(f'{feat_prefix}_p{i:02d}')

    # EVM 집계 피처
    expected_order.extend(['activity_mean_evm', 'activity_std_evm', 'activity_max_evm'])

    # 존재하는 컬럼만 선택 (순서 유지)
    feature_cols = [c for c in expected_order if c in df.columns]

    # frame_idx 추가
    df = df[['frame_idx'] + feature_cols]

    # 저장
    output_path = output_dir / 'features.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved features to {output_path}")

    # 요약
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"  - Original histogram: {len(HISTOGRAM_FEATURES) * N_BINS}")
    print(f"  - Original aggregates: 3")
    if use_evm:
        print(f"  - EVM histogram: {len(HISTOGRAM_FEATURES) * N_BINS}")
        print(f"  - EVM aggregates: 3")
    print("=" * 60)

    return df


def main():
    parser = argparse.ArgumentParser(description="Create prediction features from optical flow")
    parser.add_argument(
        "--optical_flow_dir",
        type=str,
        required=True,
        help="Directory containing original optical flow JSON files"
    )
    parser.add_argument(
        "--optical_flow_evm_dir",
        type=str,
        default=None,
        help="Directory containing EVM optical flow JSON files (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for features"
    )
    parser.add_argument(
        "--bin_edges_path",
        type=str,
        default="output/D05/bin_edges.json",
        help="Path to bin edges JSON from training"
    )

    args = parser.parse_args()

    optical_flow_dir = Path(args.optical_flow_dir)
    optical_flow_evm_dir = Path(args.optical_flow_evm_dir) if args.optical_flow_evm_dir else None
    output_dir = Path(args.output_dir)
    bin_edges_path = Path(args.bin_edges_path)

    if optical_flow_evm_dir:
        create_combined_features(
            optical_flow_dir=optical_flow_dir,
            optical_flow_evm_dir=optical_flow_evm_dir,
            output_dir=output_dir,
            bin_edges_path=bin_edges_path,
        )
    else:
        # 원본만 사용
        create_combined_features(
            optical_flow_dir=optical_flow_dir,
            optical_flow_evm_dir=Path("/nonexistent"),  # 빈 경로
            output_dir=output_dir,
            bin_edges_path=bin_edges_path,
        )


if __name__ == "__main__":
    main()
