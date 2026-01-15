#!/usr/bin/env python3
"""
LightGBM 피처 생성 스크립트
- 분위수 기반 100개 구간 히스토그램 피처 생성
- 5개 통계값 × 100 구간 = 500개 피처
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any
import argparse


# 히스토그램으로 변환할 통계값들
HISTOGRAM_FEATURES = [
    'mean_magnitude',
    'std_magnitude',
    'max_magnitude',
    'area',
    'direction_uniformity'
]

# 피처명 접두사
FEATURE_PREFIXES = {
    'mean_magnitude': 'mag_mean',
    'std_magnitude': 'mag_std',
    'max_magnitude': 'mag_max',
    'area': 'area',
    'direction_uniformity': 'dir_uni'
}

N_BINS = 100


def load_frame_statistics(frames_dir: Path) -> List[Dict]:
    """frames 디렉토리에서 개별 JSON 파일들을 로드"""
    print(f"Loading frames from: {frames_dir}")

    frame_files = sorted(frames_dir.glob('frame_*.json'))
    data = []

    for frame_file in tqdm(frame_files, desc="  Loading"):
        with open(frame_file, 'r') as f:
            frame_data = json.load(f)
            data.append(frame_data)

    print(f"  Loaded {len(data)} frames")
    return data


def collect_all_values(datasets: Dict[str, List[Dict]]) -> Dict[str, np.ndarray]:
    """모든 데이터셋에서 통계값 수집"""
    print("\nCollecting all values for percentile calculation...")

    all_values = {feat: [] for feat in HISTOGRAM_FEATURES}

    for dataset_name, frames in datasets.items():
        for frame in tqdm(frames, desc=f"  {dataset_name}"):
            # union_ids와 standalone_ids 합쳐서 처리
            all_ids = frame.get('union_ids', []) + frame.get('standalone_ids', [])
            for id_stat in all_ids:
                for feat in HISTOGRAM_FEATURES:
                    if feat in id_stat:
                        all_values[feat].append(id_stat[feat])

    # Convert to numpy arrays
    for feat in HISTOGRAM_FEATURES:
        all_values[feat] = np.array(all_values[feat])
        print(f"  {feat}: {len(all_values[feat]):,} values")

    return all_values


def compute_percentile_edges(all_values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """각 통계값의 1~99 백분위수 계산 (100개 구간 경계)"""
    print("\nComputing percentile bin edges...")

    bin_edges = {}
    percentiles = np.arange(1, 100)  # 1, 2, ..., 99

    for feat in HISTOGRAM_FEATURES:
        values = all_values[feat]
        edges = np.percentile(values, percentiles)
        bin_edges[feat] = edges
        print(f"  {feat}: min={values.min():.4f}, max={values.max():.4f}")
        print(f"    edges[0]={edges[0]:.4f}, edges[49]={edges[49]:.4f}, edges[98]={edges[98]:.4f}")

    return bin_edges


def create_percentile_histogram(values: List[float], bin_edges: np.ndarray) -> np.ndarray:
    """
    분위수 구간별 비율 계산

    Args:
        values: ID별 통계값 리스트
        bin_edges: 99개 분위수 경계값

    Returns:
        100개 구간의 비율 벡터
    """
    if len(values) == 0:
        return np.zeros(N_BINS)

    values = np.array(values)

    # np.digitize: 각 값이 속하는 구간 인덱스 (0~99)
    bin_indices = np.digitize(values, bin_edges)

    # 각 구간별 개수 세기
    hist = np.bincount(bin_indices, minlength=N_BINS)

    # 비율로 정규화
    return hist / len(values)


def create_frame_features(frame: Dict, bin_edges: Dict[str, np.ndarray]) -> Dict[str, float]:
    """프레임 하나에 대한 모든 피처 생성"""
    features = {}
    # union_ids와 standalone_ids 합쳐서 처리
    ids = frame.get('union_ids', []) + frame.get('standalone_ids', [])

    # 히스토그램 피처 생성
    for feat in HISTOGRAM_FEATURES:
        values = [id_stat[feat] for id_stat in ids if feat in id_stat]
        hist = create_percentile_histogram(values, bin_edges[feat])

        prefix = FEATURE_PREFIXES[feat]
        for i, ratio in enumerate(hist):
            features[f'{prefix}_p{i:02d}'] = ratio

    # 추가 집계 피처
    features['num_ids'] = len(ids)

    if ids:
        magnitudes = [id_stat['mean_magnitude'] for id_stat in ids]
        areas = [id_stat['area'] for id_stat in ids]

        features['activity_mean'] = np.mean(magnitudes)
        features['activity_std'] = np.std(magnitudes)
        features['activity_max'] = np.max(magnitudes)
        features['total_area'] = np.sum(areas)
        features['mean_area'] = np.mean(areas)
    else:
        features['activity_mean'] = 0.0
        features['activity_std'] = 0.0
        features['activity_max'] = 0.0
        features['total_area'] = 0.0
        features['mean_area'] = 0.0

    return features


def process_dataset(frames: List[Dict], bin_edges: Dict[str, np.ndarray],
                    dataset_name: str, label: int) -> pd.DataFrame:
    """데이터셋 전체를 피처 DataFrame으로 변환"""
    print(f"\nProcessing {dataset_name} dataset...")

    all_features = []

    for frame in tqdm(frames, desc=f"  Creating features"):
        features = create_frame_features(frame, bin_edges)
        features['frame_idx'] = frame['frame_idx']
        features['label'] = label
        all_features.append(features)

    df = pd.DataFrame(all_features)
    print(f"  Created {len(df)} samples with {len(df.columns)} features")

    return df


def main():
    parser = argparse.ArgumentParser(description="Create LightGBM features from optical flow statistics")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/sanghyun/robotware/svi/output/optical_flow_per_id",
        help="Input directory with statistics.json files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sanghyun/robotware/svi/output/features",
        help="Output directory for feature files"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LightGBM Feature Creation")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Histogram features: {HISTOGRAM_FEATURES}")
    print(f"Bins: {N_BINS}")
    print("=" * 60)

    # 1. 데이터 로드 (frames 디렉토리에서 개별 JSON 파일들)
    datasets = {}
    for dataset_name in ['normal', 'sick']:
        frames_dir = input_dir / dataset_name / 'frames'
        if frames_dir.exists():
            datasets[dataset_name] = load_frame_statistics(frames_dir)

    if not datasets:
        print("No datasets found!")
        return

    # 2. 전체 데이터에서 분위수 경계 계산
    all_values = collect_all_values(datasets)
    bin_edges = compute_percentile_edges(all_values)

    # 분위수 경계 저장
    bin_edges_serializable = {k: v.tolist() for k, v in bin_edges.items()}
    with open(output_dir / 'bin_edges.json', 'w') as f:
        json.dump(bin_edges_serializable, f, indent=2)
    print(f"\nSaved bin edges to {output_dir / 'bin_edges.json'}")

    # 3. 각 데이터셋 피처 생성
    labels = {'normal': 0, 'sick': 1}
    dfs = []

    for dataset_name, frames in datasets.items():
        df = process_dataset(frames, bin_edges, dataset_name, labels[dataset_name])
        df['dataset'] = dataset_name
        dfs.append(df)

        # 개별 저장
        df.to_csv(output_dir / f'{dataset_name}_features.csv', index=False)
        print(f"  Saved to {output_dir / f'{dataset_name}_features.csv'}")

    # 4. 합쳐서 저장
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_dir / 'combined_features.csv', index=False)
    print(f"\nSaved combined features to {output_dir / 'combined_features.csv'}")

    # 5. 피처명 저장
    feature_names = [col for col in combined_df.columns if col not in ['frame_idx', 'label', 'dataset']]
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    # 요약
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(combined_df)}")
    print(f"  - Normal: {len(combined_df[combined_df['label'] == 0])}")
    print(f"  - Sick: {len(combined_df[combined_df['label'] == 1])}")
    print(f"Total features: {len(feature_names)}")
    print(f"  - Histogram features: {len(HISTOGRAM_FEATURES) * N_BINS}")
    print(f"  - Additional features: {len(feature_names) - len(HISTOGRAM_FEATURES) * N_BINS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
