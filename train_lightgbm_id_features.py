#!/usr/bin/env python3
"""
LightGBM 모델 학습 스크립트
- Normal vs Sick 분류
- 교차 검증 및 피처 중요도 분석
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import argparse


def load_data(features_path: Path, exclude_area_count: bool = True):
    """피처 데이터 로드

    Args:
        features_path: 피처 CSV 파일 경로
        exclude_area_count: 면적 및 개체수 관련 피처 제외 여부 (기본: True)
    """
    print(f"Loading data from {features_path}")
    df = pd.read_csv(features_path)

    # 피처와 라벨 분리
    exclude_cols = ['frame_idx', 'label', 'dataset']

    # 면적 및 개체수 관련 피처 제외
    if exclude_area_count:
        area_count_cols = [col for col in df.columns
                          if col.startswith('area_') or col in ['total_area', 'mean_area', 'num_ids']]
        exclude_cols.extend(area_count_cols)
        print(f"  Excluding {len(area_count_cols)} area/count features")

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y = df['label'].values

    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class distribution: Normal={sum(y==0)}, Sick={sum(y==1)}")

    return X, y, feature_cols, df


def train_single_model(X_train, y_train, X_val, y_val, params: dict):
    """단일 모델 학습"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    return model


def evaluate_model(model, X, y, dataset_name: str = "Test"):
    """모델 평가"""
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc': roc_auc_score(y, y_pred_proba)
    }

    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")

    cm = confusion_matrix(y, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Normal  Sick")
    print(f"  Actual Normal  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"  Actual Sick    {cm[1,0]:5d}  {cm[1,1]:5d}")

    return metrics, y_pred, y_pred_proba


def cross_validate(X, y, params: dict, n_splits: int = 5):
    """Stratified K-Fold 교차 검증"""
    print(f"\n{'='*60}")
    print(f"Cross Validation ({n_splits}-Fold)")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
    }
    feature_importances = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = train_single_model(X_train, y_train, X_val, y_val, params)
        metrics, _, _ = evaluate_model(model, X_val, y_val, f"Fold {fold+1}")

        for key in cv_results:
            cv_results[key].append(metrics[key])

        feature_importances.append(model.feature_importance(importance_type='gain'))

    # CV 결과 요약
    print(f"\n{'='*60}")
    print("Cross Validation Summary")
    print(f"{'='*60}")
    for key in cv_results:
        mean_val = np.mean(cv_results[key])
        std_val = np.std(cv_results[key])
        print(f"  {key.upper():10s}: {mean_val:.4f} (+/- {std_val:.4f})")

    # 평균 피처 중요도
    avg_importance = np.mean(feature_importances, axis=0)

    return cv_results, avg_importance


def plot_feature_importance(feature_names, importance, output_path: Path, top_n: int = 30):
    """피처 중요도 시각화"""
    # 상위 N개 피처
    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 12))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance (Gain)')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved feature importance plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument(
        "--features_path",
        type=str,
        default="/home/sanghyun/robotware/svi/output/features/combined_features.csv",
        help="Path to features CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sanghyun/robotware/svi/output/models",
        help="Output directory for models"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set ratio"
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--include_area_count",
        action="store_true",
        help="Include area and count features (default: exclude)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LightGBM Training")
    print("=" * 60)

    # 1. 데이터 로드
    exclude_area_count = not args.include_area_count
    X, y, feature_names, df = load_data(Path(args.features_path), exclude_area_count=exclude_area_count)

    # 2. Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # 3. 하이퍼파라미터
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }

    print(f"\nHyperparameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # 4. 교차 검증
    cv_results, avg_importance = cross_validate(X_train, y_train, params, args.cv_folds)

    # 5. 최종 모델 학습 (전체 훈련 데이터)
    print(f"\n{'='*60}")
    print("Training Final Model")
    print(f"{'='*60}")

    # Train/Val 분할 (early stopping용)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    final_model = train_single_model(X_tr, y_tr, X_val, y_val, params)

    # 6. 테스트셋 평가
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")
    test_metrics, y_pred, y_pred_proba = evaluate_model(final_model, X_test, y_test, "Test Set")

    # 7. 모델 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"lightgbm_model_{timestamp}.txt"
    final_model.save_model(str(model_path))
    print(f"\nSaved model to {model_path}")

    # 8. 결과 저장
    results = {
        'timestamp': timestamp,
        'params': params,
        'cv_results': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                       for k, v in cv_results.items()},
        'test_results': {k: float(v) for k, v in test_metrics.items()},
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_names)
    }

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    # 9. 피처 중요도 저장
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    importance_path = output_dir / f"feature_importance_{timestamp}.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importance to {importance_path}")

    # 10. 피처 중요도 시각화
    plot_path = output_dir / f"feature_importance_{timestamp}.png"
    plot_feature_importance(feature_names, final_model.feature_importance(importance_type='gain'), plot_path)

    # 11. 상위 피처 출력
    print(f"\n{'='*60}")
    print("Top 20 Important Features")
    print(f"{'='*60}")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.2f}")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
