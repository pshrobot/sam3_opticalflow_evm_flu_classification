#!/usr/bin/env python3
"""
LightGBM 예측 스크립트
- 학습된 모델로 예측 수행
- 프레임별 예측 확률 출력
- 결과 요약
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
import argparse


def load_model(model_path: Path) -> lgb.Booster:
    """LightGBM 모델 로드"""
    print(f"Loading model from: {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    print(f"  Model loaded successfully")
    print(f"  Number of features: {model.num_feature()}")
    return model


def load_features(features_path: Path) -> tuple:
    """피처 파일 로드"""
    print(f"Loading features from: {features_path}")
    df = pd.read_csv(features_path)

    # frame_idx 분리
    frame_idx = df['frame_idx'].values if 'frame_idx' in df.columns else None

    # 피처만 추출 (frame_idx, label, dataset 제외)
    exclude_cols = ['frame_idx', 'label', 'dataset']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values

    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(feature_cols)}")

    return X, frame_idx, feature_cols


def predict(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """예측 수행"""
    y_pred_proba = model.predict(X)
    return y_pred_proba


def main():
    parser = argparse.ArgumentParser(description="Predict using LightGBM model")
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to features CSV"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LightGBM Prediction")
    print("=" * 60)

    # 1. 모델 로드
    model = load_model(Path(args.model_path))

    # 2. 피처 로드
    X, frame_idx, feature_cols = load_features(Path(args.features_path))

    # 3. 예측
    print("\nPredicting...")
    y_pred_proba = predict(model, X)
    y_pred = (y_pred_proba > args.threshold).astype(int)

    # 4. 결과 분석
    n_sick = np.sum(y_pred == 1)
    n_normal = np.sum(y_pred == 0)
    sick_ratio = n_sick / len(y_pred) * 100

    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Total frames: {len(y_pred)}")
    print(f"Predicted Sick: {n_sick} ({sick_ratio:.1f}%)")
    print(f"Predicted Normal: {n_normal} ({100-sick_ratio:.1f}%)")
    print(f"\nProbability Statistics:")
    print(f"  Mean: {np.mean(y_pred_proba):.4f}")
    print(f"  Std:  {np.std(y_pred_proba):.4f}")
    print(f"  Min:  {np.min(y_pred_proba):.4f}")
    print(f"  Max:  {np.max(y_pred_proba):.4f}")
    print(f"  Median: {np.median(y_pred_proba):.4f}")

    # 5. 결과 저장
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(args.model_path),
        'features_path': str(args.features_path),
        'threshold': args.threshold,
        'total_frames': len(y_pred),
        'predicted_sick': int(n_sick),
        'predicted_normal': int(n_normal),
        'sick_ratio': float(sick_ratio),
        'probability_stats': {
            'mean': float(np.mean(y_pred_proba)),
            'std': float(np.std(y_pred_proba)),
            'min': float(np.min(y_pred_proba)),
            'max': float(np.max(y_pred_proba)),
            'median': float(np.median(y_pred_proba)),
        }
    }

    # 결과 JSON 저장
    results_path = output_dir / 'prediction_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # 프레임별 예측 CSV 저장
    df_pred = pd.DataFrame({
        'frame_idx': frame_idx if frame_idx is not None else range(len(y_pred)),
        'probability': y_pred_proba,
        'prediction': y_pred,
        'label': ['sick' if p == 1 else 'normal' for p in y_pred]
    })
    predictions_path = output_dir / 'frame_predictions.csv'
    df_pred.to_csv(predictions_path, index=False)
    print(f"Saved frame predictions to {predictions_path}")

    # 6. 최종 판정
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")

    if sick_ratio > 70:
        verdict = "SICK (HIGH CONFIDENCE)"
    elif sick_ratio > 50:
        verdict = "SICK (MODERATE CONFIDENCE)"
    elif sick_ratio > 30:
        verdict = "UNCERTAIN - Further inspection recommended"
    else:
        verdict = "NORMAL"

    print(f"  {verdict}")
    print(f"  (Based on {sick_ratio:.1f}% of frames predicted as sick)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
