#!/usr/bin/env python3
"""
Per-ID Optical Flow Statistics Extraction Script
- 마스크 이미지의 ID값(개별 닭)을 기반으로 ID별 optical flow 계산
- 이전/현재 프레임에서 겹치는 ID는 합집합 마스크 사용
- 예외: bbox IoU가 0.5 이상인 세그멘트는 단독 처리
- 모든 ID를 처리 (BBox 크기 제한 없음)
"""

import os
import sys
import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from datetime import datetime


def compute_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple:
    """
    Compute optical flow using Farneback method.

    Returns:
        flow: (H, W, 2) flow vectors
        magnitude: (H, W) flow magnitude
        angle: (H, W) flow angle in radians
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    return flow, magnitude, angle


def get_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """마스크에서 BBox 계산 (x, y, w, h)"""
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0:
        return (0, 0, 0, 0)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


def compute_bbox_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    두 BBox의 IoU (Intersection over Union) 계산

    Args:
        bbox1, bbox2: (x, y, w, h) 형식의 bounding box

    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 빈 bbox 처리
    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        return 0.0

    # 교집합 영역 계산
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 합집합 영역 계산
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_mask_overlap(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    """두 마스크가 겹치는지 확인"""
    return np.any(mask1 & mask2)


def get_ids_from_mask(mask: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """
    마스크에서 각 ID별로 개별 마스크 추출

    Args:
        mask: ID 마스크 (각 픽셀 값이 ID)

    Returns:
        Dict: {id: {'mask': boolean_mask, 'bbox': (x, y, w, h)}}
    """
    result = {}

    # 고유 ID 목록 (배경 0 제외)
    ids = np.unique(mask)
    ids = ids[ids > 0]

    for id_val in ids:
        id_mask = (mask == id_val)
        bbox = get_bbox(id_mask)

        result[int(id_val)] = {
            'mask': id_mask,
            'bbox': bbox
        }

    return result


def extract_id_statistics(magnitude: np.ndarray, angle: np.ndarray,
                          id_mask: np.ndarray, id_val: int) -> Dict[str, Any]:
    """
    Extract optical flow statistics for a single ID.

    Args:
        magnitude: Flow magnitude array
        angle: Flow angle array (radians)
        id_mask: Boolean mask for this ID
        id_val: ID value

    Returns:
        Statistics dictionary for this ID
    """
    valid_mag = magnitude[id_mask]
    valid_angle = angle[id_mask]

    if len(valid_mag) == 0:
        return None

    # Centroid calculation
    y_coords, x_coords = np.where(id_mask)
    centroid = [float(np.mean(x_coords)), float(np.mean(y_coords))]

    # Direction statistics (circular statistics)
    sin_sum = np.sum(np.sin(valid_angle))
    cos_sum = np.sum(np.cos(valid_angle))
    mean_direction = np.arctan2(sin_sum, cos_sum)
    resultant_length = np.sqrt(sin_sum**2 + cos_sum**2) / len(valid_angle)

    return {
        'id': int(id_val),
        'area': int(np.sum(id_mask)),
        'centroid': centroid,
        'mean_magnitude': float(np.mean(valid_mag)),
        'std_magnitude': float(np.std(valid_mag)),
        'max_magnitude': float(np.max(valid_mag)),
        'median_magnitude': float(np.median(valid_mag)),
        'mean_direction': float(mean_direction),
        'direction_uniformity': float(resultant_length),
    }


def compute_frame_summary(id_stats: List[Dict]) -> Dict[str, Any]:
    """
    Compute summary statistics for all IDs in a frame.
    """
    if not id_stats:
        return {
            'total_ids': 0,
            'mean_activity': 0.0,
            'std_activity': 0.0,
            'median_activity': 0.0,
            'num_high_activity': 0,
            'num_low_activity': 0,
        }

    activities = [s['mean_magnitude'] for s in id_stats]
    mean_activity = np.mean(activities)
    std_activity = np.std(activities)

    # Identify high/low activity IDs (> 1.5 std from mean)
    if std_activity > 0:
        threshold_high = mean_activity + 1.5 * std_activity
        threshold_low = mean_activity - 1.5 * std_activity
        num_high = sum(1 for a in activities if a > threshold_high)
        num_low = sum(1 for a in activities if a < threshold_low and a > 0)
    else:
        num_high = 0
        num_low = 0

    return {
        'total_ids': len(id_stats),
        'mean_activity': float(mean_activity),
        'std_activity': float(std_activity),
        'median_activity': float(np.median(activities)),
        'num_high_activity': num_high,
        'num_low_activity': num_low,
    }


def get_frame_mask_pairs(frames_dir: Path, masks_dir: Path) -> List[Dict[str, Path]]:
    """Get matching frame and mask file pairs."""
    frames = sorted(frames_dir.glob('*.jpg'))
    pairs = []

    for frame_path in frames:
        # mask has same name but .png extension
        mask_name = frame_path.stem + '.png'
        mask_path = masks_dir / mask_name

        if mask_path.exists():
            pairs.append({
                'frame': frame_path,
                'mask': mask_path
            })

    return pairs


def process_dataset(
    dataset_name: str,
    frames_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    checkpoint_interval: int = 500,
) -> Dict[str, Any]:
    """
    Process a single dataset and extract per-ID optical flow statistics.
    Each frame's result is saved as a separate JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_output_dir = output_dir / 'frames'
    frames_output_dir.mkdir(parents=True, exist_ok=True)

    # Get frame-mask pairs
    pairs = get_frame_mask_pairs(frames_dir, masks_dir)
    total_pairs = len(pairs)

    if total_pairs < 2:
        print(f"Not enough frames for {dataset_name}: {total_pairs}")
        return {}

    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"Frames: {total_pairs}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Check for checkpoint (last processed frame index)
    checkpoint_path = output_dir / 'checkpoint.txt'
    start_idx = 1

    if checkpoint_path.exists():
        try:
            start_idx = int(checkpoint_path.read_text().strip()) + 1
            print(f"Resuming from frame {start_idx}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            start_idx = 1

    # Collect summary stats for overall summary
    all_activities = []
    all_id_counts = []
    processed_count = 0

    # Process frame pairs
    pbar = tqdm(range(start_idx, total_pairs), desc=f"Processing {dataset_name}")

    for i in pbar:
        prev_pair = pairs[i - 1]
        curr_pair = pairs[i]

        # Load frames
        prev_frame = cv2.imread(str(prev_pair['frame']))
        curr_frame = cv2.imread(str(curr_pair['frame']))

        if prev_frame is None or curr_frame is None:
            continue

        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Load masks (both previous and current frames)
        prev_mask = cv2.imread(str(prev_pair['mask']), cv2.IMREAD_UNCHANGED)
        curr_mask = cv2.imread(str(curr_pair['mask']), cv2.IMREAD_UNCHANGED)

        if prev_mask is None or curr_mask is None:
            continue

        # Compute optical flow
        flow, magnitude, angle = compute_optical_flow(prev_gray, curr_gray)

        # 각 프레임에서 ID별 마스크 추출
        prev_ids_info = get_ids_from_mask(prev_mask)
        curr_ids_info = get_ids_from_mask(curr_mask)

        # 공통 ID와 단독 ID 분류
        prev_only_ids = set(prev_ids_info.keys()) - set(curr_ids_info.keys())
        curr_only_ids = set(curr_ids_info.keys()) - set(prev_ids_info.keys())
        common_ids = set(prev_ids_info.keys()) & set(curr_ids_info.keys())

        id_stats = []
        union_stats = []  # 합집합으로 처리된 ID들
        standalone_stats = []  # 단독으로 처리된 ID들

        # 공통 ID 처리: bbox IoU >= 0.5면 단독, 그렇지 않고 겹치면 합집합
        for id_val in common_ids:
            prev_info = prev_ids_info[id_val]
            curr_info = curr_ids_info[id_val]

            bbox_iou = compute_bbox_iou(prev_info['bbox'], curr_info['bbox'])

            if bbox_iou >= 0.5:
                # bbox IoU >= 0.5: 각각 단독 처리
                stats_prev = extract_id_statistics(magnitude, angle, prev_info['mask'], id_val)
                if stats_prev is not None:
                    stats_prev['bbox'] = prev_info['bbox']
                    stats_prev['frame_type'] = 'prev'
                    stats_prev['bbox_iou'] = bbox_iou
                    stats_prev['process_type'] = 'standalone_high_iou'
                    standalone_stats.append(stats_prev)

                stats_curr = extract_id_statistics(magnitude, angle, curr_info['mask'], id_val)
                if stats_curr is not None:
                    stats_curr['bbox'] = curr_info['bbox']
                    stats_curr['frame_type'] = 'curr'
                    stats_curr['bbox_iou'] = bbox_iou
                    stats_curr['process_type'] = 'standalone_high_iou'
                    standalone_stats.append(stats_curr)
            else:
                # bbox IoU < 0.5: 겹침 여부 확인 후 합집합 또는 단독 처리
                has_overlap = compute_mask_overlap(prev_info['mask'], curr_info['mask'])

                if has_overlap:
                    # 마스크 겹침 있음: 합집합 사용
                    union_mask = prev_info['mask'] | curr_info['mask']
                    union_bbox = get_bbox(union_mask)

                    stats = extract_id_statistics(magnitude, angle, union_mask, id_val)
                    if stats is not None:
                        stats['bbox'] = union_bbox
                        stats['frame_type'] = 'union'
                        stats['bbox_iou'] = bbox_iou
                        stats['process_type'] = 'union'
                        stats['prev_bbox'] = prev_info['bbox']
                        stats['curr_bbox'] = curr_info['bbox']
                        union_stats.append(stats)
                else:
                    # 마스크 겹침 없음: 각각 단독 처리
                    stats_prev = extract_id_statistics(magnitude, angle, prev_info['mask'], id_val)
                    if stats_prev is not None:
                        stats_prev['bbox'] = prev_info['bbox']
                        stats_prev['frame_type'] = 'prev'
                        stats_prev['bbox_iou'] = bbox_iou
                        stats_prev['process_type'] = 'standalone_no_overlap'
                        standalone_stats.append(stats_prev)

                    stats_curr = extract_id_statistics(magnitude, angle, curr_info['mask'], id_val)
                    if stats_curr is not None:
                        stats_curr['bbox'] = curr_info['bbox']
                        stats_curr['frame_type'] = 'curr'
                        stats_curr['bbox_iou'] = bbox_iou
                        stats_curr['process_type'] = 'standalone_no_overlap'
                        standalone_stats.append(stats_curr)

        # 이전 프레임에만 있는 ID들 (단독 처리)
        for id_val in prev_only_ids:
            info = prev_ids_info[id_val]
            stats = extract_id_statistics(magnitude, angle, info['mask'], id_val)
            if stats is not None:
                stats['bbox'] = info['bbox']
                stats['frame_type'] = 'prev'
                stats['process_type'] = 'standalone_prev_only'
                standalone_stats.append(stats)

        # 현재 프레임에만 있는 ID들 (단독 처리)
        for id_val in curr_only_ids:
            info = curr_ids_info[id_val]
            stats = extract_id_statistics(magnitude, angle, info['mask'], id_val)
            if stats is not None:
                stats['bbox'] = info['bbox']
                stats['frame_type'] = 'curr'
                stats['process_type'] = 'standalone_curr_only'
                standalone_stats.append(stats)

        # 전체 통계 합산
        id_stats = union_stats + standalone_stats

        # Compute frame summary
        summary = compute_frame_summary(id_stats)

        # Build frame statistics
        frame_stats = {
            'frame_idx': i,
            'prev_frame': prev_pair['frame'].name,
            'curr_frame': curr_pair['frame'].name,
            'num_ids': len(id_stats),
            'num_union': len(union_stats),
            'num_standalone': len(standalone_stats),
            'num_common_ids': len(common_ids),
            'num_prev_only_ids': len(prev_only_ids),
            'num_curr_only_ids': len(curr_only_ids),
            'union_ids': union_stats,
            'standalone_ids': standalone_stats,
            'summary': summary,
        }

        # Save frame result to individual file
        frame_output_path = frames_output_dir / f'frame_{i:06d}.json'
        with open(frame_output_path, 'w') as f:
            json.dump(frame_stats, f)

        # Collect for overall summary
        all_activities.append(summary['mean_activity'])
        all_id_counts.append(len(id_stats))
        processed_count += 1

        # Update progress bar
        pbar.set_postfix({
            'union': len(union_stats),
            'stand': len(standalone_stats),
            'activity': f'{summary["mean_activity"]:.2f}'
        })

        # Save checkpoint
        if i % checkpoint_interval == 0:
            checkpoint_path.write_text(str(i))

    pbar.close()

    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Compute overall summary
    if processed_count > 0:
        overall_summary = {
            'dataset': dataset_name,
            'total_frames': total_pairs,
            'processed_pairs': processed_count,
            'activity': {
                'mean': float(np.mean(all_activities)),
                'std': float(np.std(all_activities)),
                'min': float(np.min(all_activities)),
                'max': float(np.max(all_activities)),
            },
            'id_count': {
                'mean': float(np.mean(all_id_counts)),
                'std': float(np.std(all_id_counts)),
                'min': int(np.min(all_id_counts)),
                'max': int(np.max(all_id_counts)),
            },
            'timestamp': datetime.now().isoformat()
        }

        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(overall_summary, f, indent=2)

        print(f"\nSummary for {dataset_name}:")
        print(f"  - Processed pairs: {processed_count}")
        print(f"  - Mean IDs per frame: {overall_summary['id_count']['mean']:.1f}")
        print(f"  - Mean activity: {overall_summary['activity']['mean']:.4f}")
        print(f"  - Output frames: {frames_output_dir}")

        return overall_summary

    return {}


def main():
    parser = argparse.ArgumentParser(description="Extract Per-ID Optical Flow Statistics")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sanghyun/robotware/svi/data",
        help="Data directory"
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default="/home/sanghyun/robotware/svi/output/sam3_masks",
        help="Masks directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sanghyun/robotware/svi/output/optical_flow_per_id",
        help="Output directory"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["normal", "sick"],
        help="Datasets to process"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=500,
        help="Save checkpoint every N frames"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Per-ID Optical Flow Statistics Extraction")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Masks directory: {args.masks_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets: {args.datasets}")
    print("=" * 60)

    # Process datasets
    data_dir = Path(args.data_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)

    all_summaries = {}

    for dataset in args.datasets:
        frames_dir = data_dir / dataset / "frames"
        dataset_masks_dir = masks_dir / dataset / "mask"
        dataset_output_dir = output_dir / dataset

        if not frames_dir.exists():
            print(f"Frames directory not found: {frames_dir}")
            continue

        if not dataset_masks_dir.exists():
            print(f"Masks directory not found: {dataset_masks_dir}")
            continue

        summary = process_dataset(
            dataset_name=dataset,
            frames_dir=frames_dir,
            masks_dir=dataset_masks_dir,
            output_dir=dataset_output_dir,
            checkpoint_interval=args.checkpoint_interval,
        )

        all_summaries[dataset] = summary

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)

    for dataset, summary in all_summaries.items():
        if summary:
            print(f"\n{dataset.upper()}:")
            print(f"  - Processed pairs: {summary.get('processed_pairs', 0)}")
            print(f"  - Mean IDs/frame: {summary.get('id_count', {}).get('mean', 0):.1f}")
            print(f"  - Mean activity: {summary.get('activity', {}).get('mean', 0):.4f}")
            print(f"  - Output: {output_dir / dataset}")

    print("=" * 60)


if __name__ == "__main__":
    main()
