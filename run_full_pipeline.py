#!/usr/bin/env python3
"""
전체 파이프라인 배치 스크립트
- 비디오 하나에 대해 프레임 추출 → SAM3 → EVM → Optical Flow → 피처 생성 → 예측까지 수행
- 백그라운드 실행 지원 (nohup)
- 체크포인트 지원 (중단 후 재개)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))


def log(msg):
    """타임스탬프 포함 로그 출력"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def check_step_completed(output_dir: Path, step_name: str) -> bool:
    """단계 완료 여부 확인"""
    marker_file = output_dir / f".{step_name}_completed"
    return marker_file.exists()


def mark_step_completed(output_dir: Path, step_name: str):
    """단계 완료 마커 생성"""
    marker_file = output_dir / f".{step_name}_completed"
    marker_file.write_text(datetime.now().isoformat())


def step1_extract_frames(video_path: Path, output_dir: Path) -> bool:
    """Step 1: 프레임 추출"""
    log("=" * 60)
    log("STEP 1: Frame Extraction")
    log("=" * 60)

    frames_dir = output_dir / "frames"

    if check_step_completed(output_dir, "step1"):
        log(f"Step 1 already completed. Skipping...")
        return True

    from extract_video_frames import extract_frames

    try:
        extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            fps=1.0
        )
        mark_step_completed(output_dir, "step1")
        log("Step 1 completed!")
        return True
    except Exception as e:
        log(f"Step 1 failed: {e}")
        return False


def step2_extract_sam3_masks(output_dir: Path) -> bool:
    """Step 2: SAM3 마스크 추출"""
    log("=" * 60)
    log("STEP 2: SAM3 Mask Extraction")
    log("=" * 60)

    if check_step_completed(output_dir, "step2"):
        log(f"Step 2 already completed. Skipping...")
        return True

    frames_dir = output_dir / "frames"
    sam3_output_dir = output_dir / "sam3_masks"

    from extract_sam3_masks import SAM3BatchProcessor, process_frames_directory, get_color_map
    import numpy as np

    try:
        processor = SAM3BatchProcessor(
            text_prompt="chicken",
            threshold=0.1,
            device="cuda",
            batch_size=8,
        )
        colors = get_color_map(256)

        process_frames_directory(
            frames_dir=frames_dir,
            output_dir=sam3_output_dir,
            processor=processor,
            colors=colors,
            batch_size=8,
        )

        mark_step_completed(output_dir, "step2")
        log("Step 2 completed!")
        return True
    except Exception as e:
        log(f"Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step3_apply_evm(video_path: Path, output_dir: Path) -> bool:
    """Step 3: EVM 적용"""
    log("=" * 60)
    log("STEP 3: EVM Application")
    log("=" * 60)

    if check_step_completed(output_dir, "step3"):
        log(f"Step 3 already completed. Skipping...")
        return True

    evm_output_dir = output_dir / "evm_frames"

    from apply_evm import apply_evm_to_video

    try:
        apply_evm_to_video(
            video_path=video_path,
            output_dir=evm_output_dir,
            freq_min=0.5,
            freq_max=2.0,
            amplification=20,
        )
        mark_step_completed(output_dir, "step3")
        log("Step 3 completed!")
        return True
    except Exception as e:
        log(f"Step 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step4_extract_optical_flow(output_dir: Path) -> bool:
    """Step 4: Optical Flow 추출 (원본)"""
    log("=" * 60)
    log("STEP 4: Optical Flow Extraction (Original)")
    log("=" * 60)

    if check_step_completed(output_dir, "step4"):
        log(f"Step 4 already completed. Skipping...")
        return True

    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "sam3_masks" / "mask"
    optical_flow_dir = output_dir / "optical_flow"

    from extract_optical_flow_per_id import process_dataset

    try:
        process_dataset(
            dataset_name="prediction",
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            output_dir=optical_flow_dir,
            checkpoint_interval=500,
        )
        mark_step_completed(output_dir, "step4")
        log("Step 4 completed!")
        return True
    except Exception as e:
        log(f"Step 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step5_extract_optical_flow_evm(output_dir: Path) -> bool:
    """Step 5: Optical Flow 추출 (EVM)"""
    log("=" * 60)
    log("STEP 5: Optical Flow Extraction (EVM)")
    log("=" * 60)

    if check_step_completed(output_dir, "step5"):
        log(f"Step 5 already completed. Skipping...")
        return True

    evm_dir = output_dir / "evm_frames"
    masks_dir = output_dir / "sam3_masks" / "mask"
    optical_flow_evm_dir = output_dir / "optical_flow_evm"

    from extract_optical_flow_evm import process_dataset

    try:
        process_dataset(
            dataset_name="prediction_evm",
            evm_dir=evm_dir,
            masks_dir=masks_dir,
            output_dir=optical_flow_evm_dir,
            checkpoint_interval=500,
        )
        mark_step_completed(output_dir, "step5")
        log("Step 5 completed!")
        return True
    except Exception as e:
        log(f"Step 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step6_create_features(output_dir: Path, bin_edges_path: Path) -> bool:
    """Step 6: LightGBM 피처 생성"""
    log("=" * 60)
    log("STEP 6: Feature Generation")
    log("=" * 60)

    if check_step_completed(output_dir, "step6"):
        log(f"Step 6 already completed. Skipping...")
        return True

    optical_flow_dir = output_dir / "optical_flow" / "frames"
    optical_flow_evm_dir = output_dir / "optical_flow_evm" / "frames"
    features_dir = output_dir / "features"

    from create_lightgbm_features_predict import create_combined_features

    try:
        create_combined_features(
            optical_flow_dir=optical_flow_dir,
            optical_flow_evm_dir=optical_flow_evm_dir,
            output_dir=features_dir,
            bin_edges_path=bin_edges_path,
        )
        mark_step_completed(output_dir, "step6")
        log("Step 6 completed!")
        return True
    except Exception as e:
        log(f"Step 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step7_predict(output_dir: Path, model_path: Path) -> bool:
    """Step 7: LightGBM 예측"""
    log("=" * 60)
    log("STEP 7: Prediction")
    log("=" * 60)

    features_path = output_dir / "features" / "features.csv"
    predictions_dir = output_dir / "predictions"

    import subprocess

    try:
        cmd = [
            sys.executable,
            "predict_lightgbm.py",
            "--features_path", str(features_path),
            "--model_path", str(model_path),
            "--output_dir", str(predictions_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            log("Step 7 completed!")
            return True
        else:
            log(f"Step 7 failed with return code {result.returncode}")
            return False
    except Exception as e:
        log(f"Step 7 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline for a single video")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Output directory name (will be created under output/)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/models_combined_no_area/lightgbm_model_20260104_121820.txt",
        help="Path to trained LightGBM model"
    )
    parser.add_argument(
        "--bin_edges_path",
        type=str,
        default="output/D05/bin_edges.json",
        help="Path to bin edges JSON file"
    )
    parser.add_argument(
        "--skip_steps",
        type=str,
        default="",
        help="Comma-separated list of steps to skip (e.g., '1,2,3')"
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path("output") / args.output_name
    model_path = Path(args.model_path)
    bin_edges_path = Path(args.bin_edges_path)

    skip_steps = set(args.skip_steps.split(",")) if args.skip_steps else set()

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("FULL PIPELINE EXECUTION")
    log("=" * 60)
    log(f"Video: {video_path}")
    log(f"Output: {output_dir}")
    log(f"Model: {model_path}")
    log(f"Bin edges: {bin_edges_path}")
    log(f"Skip steps: {skip_steps or 'none'}")
    log("=" * 60)

    start_time = datetime.now()

    # 파이프라인 실행
    steps = [
        ("1", lambda: step1_extract_frames(video_path, output_dir)),
        ("2", lambda: step2_extract_sam3_masks(output_dir)),
        ("3", lambda: step3_apply_evm(video_path, output_dir)),
        ("4", lambda: step4_extract_optical_flow(output_dir)),
        ("5", lambda: step5_extract_optical_flow_evm(output_dir)),
        ("6", lambda: step6_create_features(output_dir, bin_edges_path)),
        ("7", lambda: step7_predict(output_dir, model_path)),
    ]

    for step_num, step_func in steps:
        if step_num in skip_steps:
            log(f"Skipping step {step_num} as requested")
            continue

        success = step_func()
        if not success:
            log(f"Pipeline failed at step {step_num}")
            sys.exit(1)

    elapsed = datetime.now() - start_time

    log("=" * 60)
    log("PIPELINE COMPLETED SUCCESSFULLY!")
    log(f"Total time: {elapsed}")
    log("=" * 60)

    # 최종 결과 출력
    predictions_path = output_dir / "predictions" / "prediction_results.json"
    if predictions_path.exists():
        with open(predictions_path) as f:
            results = json.load(f)

        log("\nFINAL RESULTS:")
        log(f"  Total frames: {results['total_frames']}")
        log(f"  Predicted Sick: {results['predicted_sick']} ({results['sick_ratio']:.1f}%)")
        log(f"  Predicted Normal: {results['predicted_normal']} ({100-results['sick_ratio']:.1f}%)")

        sick_ratio = results['sick_ratio']
        if sick_ratio > 70:
            verdict = "SICK (HIGH CONFIDENCE)"
        elif sick_ratio > 50:
            verdict = "SICK (MODERATE CONFIDENCE)"
        elif sick_ratio > 30:
            verdict = "UNCERTAIN"
        else:
            verdict = "NORMAL"

        log(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
