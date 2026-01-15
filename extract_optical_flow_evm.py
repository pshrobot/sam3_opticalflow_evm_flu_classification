#!/usr/bin/env python3
"""
EVM 프레임에서 Optical Flow 추출
- EVM 프레임과 기존 SAM3 마스크를 타임스탬프로 매핑
- 기존 extract_optical_flow_per_id.py와 동일한 방식으로 통계 추출
"""

import os
import cv2
import json
import numpy as np
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from datetime import datetime


def compute_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple:
    """Compute optical flow using Farneback method."""
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
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return flow, magnitude, angle


def get_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """마스크에서 BBox 계산"""
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0:
        return (0, 0, 0, 0)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


def compute_bbox_iou(bbox1, bbox2) -> float:
    """두 BBox의 IoU 계산"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        return 0.0
    inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
    inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_mask_overlap(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    return np.any(mask1 & mask2)


def get_ids_from_mask(mask: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """마스크에서 각 ID별로 개별 마스크 추출"""
    result = {}
    ids = np.unique(mask)
    ids = ids[ids > 0]
    for id_val in ids:
        id_mask = (mask == id_val)
        result[int(id_val)] = {'mask': id_mask, 'bbox': get_bbox(id_mask)}
    return result


def extract_id_statistics(magnitude, angle, id_mask, id_val) -> Dict[str, Any]:
    """Extract optical flow statistics for a single ID."""
    valid_mag = magnitude[id_mask]
    valid_angle = angle[id_mask]
    if len(valid_mag) == 0:
        return None
    y_coords, x_coords = np.where(id_mask)
    centroid = [float(np.mean(x_coords)), float(np.mean(y_coords))]
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
    """Compute summary statistics for all IDs in a frame."""
    if not id_stats:
        return {
            'total_ids': 0, 'mean_activity': 0.0, 'std_activity': 0.0,
            'median_activity': 0.0, 'num_high_activity': 0, 'num_low_activity': 0,
        }
    activities = [s['mean_magnitude'] for s in id_stats]
    mean_activity = np.mean(activities)
    std_activity = np.std(activities)
    if std_activity > 0:
        threshold_high = mean_activity + 1.5 * std_activity
        threshold_low = mean_activity - 1.5 * std_activity
        num_high = sum(1 for a in activities if a > threshold_high)
        num_low = sum(1 for a in activities if a < threshold_low and a > 0)
    else:
        num_high = num_low = 0
    return {
        'total_ids': len(id_stats),
        'mean_activity': float(mean_activity),
        'std_activity': float(std_activity),
        'median_activity': float(np.median(activities)),
        'num_high_activity': num_high,
        'num_low_activity': num_low,
    }


def parse_evm_timestamp(filename: str) -> float:
    """EVM 프레임 파일명에서 타임스탬프 추출 (frame_0.0s.jpg -> 0.0)"""
    match = re.search(r'frame_(\d+\.?\d*)s\.jpg', filename)
    return float(match.group(1)) if match else -1


def parse_mask_timestamp(filename: str) -> float:
    """마스크 파일명에서 타임스탬프 추출 (frame_0000_0.0s.png -> 0.0)"""
    match = re.search(r'frame_\d+_(\d+\.?\d*)s\.png', filename)
    return float(match.group(1)) if match else -1


def build_timestamp_mapping(evm_dir: Path, masks_dir: Path) -> List[Dict]:
    """EVM 프레임과 마스크 파일을 타임스탬프로 매핑"""
    evm_files = {parse_evm_timestamp(f.name): f for f in evm_dir.glob('*.jpg')}
    mask_files = {parse_mask_timestamp(f.name): f for f in masks_dir.glob('*.png')}

    # 공통 타임스탬프 찾기
    common_ts = sorted(set(evm_files.keys()) & set(mask_files.keys()))
    common_ts = [ts for ts in common_ts if ts >= 0]

    pairs = []
    for ts in common_ts:
        pairs.append({
            'timestamp': ts,
            'evm_frame': evm_files[ts],
            'mask': mask_files[ts]
        })

    return pairs


def process_dataset(
    dataset_name: str,
    evm_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    checkpoint_interval: int = 500,
) -> Dict[str, Any]:
    """EVM 프레임에서 optical flow 추출"""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_output_dir = output_dir / 'frames'
    frames_output_dir.mkdir(parents=True, exist_ok=True)

    # 타임스탬프 기반 매핑
    pairs = build_timestamp_mapping(evm_dir, masks_dir)
    total_pairs = len(pairs)

    if total_pairs < 2:
        print(f"Not enough matching frames for {dataset_name}: {total_pairs}")
        return {}

    print(f"\n{'='*60}")
    print(f"Processing EVM: {dataset_name}")
    print(f"EVM frames: {evm_dir}")
    print(f"Masks: {masks_dir}")
    print(f"Matched pairs: {total_pairs}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Checkpoint
    checkpoint_path = output_dir / 'checkpoint.txt'
    start_idx = 1
    if checkpoint_path.exists():
        try:
            start_idx = int(checkpoint_path.read_text().strip()) + 1
            print(f"Resuming from frame {start_idx}")
        except:
            start_idx = 1

    all_activities = []
    all_id_counts = []
    processed_count = 0

    pbar = tqdm(range(start_idx, total_pairs), desc=f"Processing {dataset_name}")

    for i in pbar:
        prev_pair = pairs[i - 1]
        curr_pair = pairs[i]

        # Load EVM frames
        prev_frame = cv2.imread(str(prev_pair['evm_frame']))
        curr_frame = cv2.imread(str(curr_pair['evm_frame']))
        if prev_frame is None or curr_frame is None:
            continue

        # EVM 프레임 크기를 마스크에 맞추기 위해 리사이즈 필요할 수 있음
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Load masks
        prev_mask = cv2.imread(str(prev_pair['mask']), cv2.IMREAD_UNCHANGED)
        curr_mask = cv2.imread(str(curr_pair['mask']), cv2.IMREAD_UNCHANGED)
        if prev_mask is None or curr_mask is None:
            continue

        # 크기 맞추기 (EVM 프레임은 원본 크기로 복원되었지만 확인)
        if prev_gray.shape != prev_mask.shape:
            prev_gray = cv2.resize(prev_gray, (prev_mask.shape[1], prev_mask.shape[0]))
            curr_gray = cv2.resize(curr_gray, (curr_mask.shape[1], curr_mask.shape[0]))

        # Compute optical flow
        flow, magnitude, angle = compute_optical_flow(prev_gray, curr_gray)

        # ID별 마스크 추출
        prev_ids_info = get_ids_from_mask(prev_mask)
        curr_ids_info = get_ids_from_mask(curr_mask)

        prev_only_ids = set(prev_ids_info.keys()) - set(curr_ids_info.keys())
        curr_only_ids = set(curr_ids_info.keys()) - set(prev_ids_info.keys())
        common_ids = set(prev_ids_info.keys()) & set(curr_ids_info.keys())

        union_stats = []
        standalone_stats = []

        # 공통 ID 처리
        for id_val in common_ids:
            prev_info = prev_ids_info[id_val]
            curr_info = curr_ids_info[id_val]
            bbox_iou = compute_bbox_iou(prev_info['bbox'], curr_info['bbox'])

            if bbox_iou >= 0.5:
                for info, frame_type in [(prev_info, 'prev'), (curr_info, 'curr')]:
                    stats = extract_id_statistics(magnitude, angle, info['mask'], id_val)
                    if stats:
                        stats.update({'bbox': info['bbox'], 'frame_type': frame_type,
                                     'bbox_iou': bbox_iou, 'process_type': 'standalone_high_iou'})
                        standalone_stats.append(stats)
            else:
                if compute_mask_overlap(prev_info['mask'], curr_info['mask']):
                    union_mask = prev_info['mask'] | curr_info['mask']
                    stats = extract_id_statistics(magnitude, angle, union_mask, id_val)
                    if stats:
                        stats.update({'bbox': get_bbox(union_mask), 'frame_type': 'union',
                                     'bbox_iou': bbox_iou, 'process_type': 'union'})
                        union_stats.append(stats)
                else:
                    for info, frame_type in [(prev_info, 'prev'), (curr_info, 'curr')]:
                        stats = extract_id_statistics(magnitude, angle, info['mask'], id_val)
                        if stats:
                            stats.update({'bbox': info['bbox'], 'frame_type': frame_type,
                                         'bbox_iou': bbox_iou, 'process_type': 'standalone_no_overlap'})
                            standalone_stats.append(stats)

        # 단독 ID 처리
        for id_val in prev_only_ids:
            info = prev_ids_info[id_val]
            stats = extract_id_statistics(magnitude, angle, info['mask'], id_val)
            if stats:
                stats.update({'bbox': info['bbox'], 'frame_type': 'prev', 'process_type': 'standalone_prev_only'})
                standalone_stats.append(stats)

        for id_val in curr_only_ids:
            info = curr_ids_info[id_val]
            stats = extract_id_statistics(magnitude, angle, info['mask'], id_val)
            if stats:
                stats.update({'bbox': info['bbox'], 'frame_type': 'curr', 'process_type': 'standalone_curr_only'})
                standalone_stats.append(stats)

        id_stats = union_stats + standalone_stats
        summary = compute_frame_summary(id_stats)

        frame_stats = {
            'frame_idx': i,
            'prev_timestamp': prev_pair['timestamp'],
            'curr_timestamp': curr_pair['timestamp'],
            'prev_frame': prev_pair['evm_frame'].name,
            'curr_frame': curr_pair['evm_frame'].name,
            'num_ids': len(id_stats),
            'num_union': len(union_stats),
            'num_standalone': len(standalone_stats),
            'union_ids': union_stats,
            'standalone_ids': standalone_stats,
            'summary': summary,
        }

        with open(frames_output_dir / f'frame_{i:06d}.json', 'w') as f:
            json.dump(frame_stats, f)

        all_activities.append(summary['mean_activity'])
        all_id_counts.append(len(id_stats))
        processed_count += 1

        pbar.set_postfix({'union': len(union_stats), 'stand': len(standalone_stats),
                          'activity': f'{summary["mean_activity"]:.2f}'})

        if i % checkpoint_interval == 0:
            checkpoint_path.write_text(str(i))

    pbar.close()
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    if processed_count > 0:
        overall_summary = {
            'dataset': dataset_name,
            'source': 'evm',
            'total_pairs': total_pairs,
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

        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(overall_summary, f, indent=2)

        print(f"\nSummary for {dataset_name} (EVM):")
        print(f"  - Processed pairs: {processed_count}")
        print(f"  - Mean IDs per frame: {overall_summary['id_count']['mean']:.1f}")
        print(f"  - Mean activity: {overall_summary['activity']['mean']:.4f}")

        return overall_summary
    return {}


def main():
    parser = argparse.ArgumentParser(description="Extract Optical Flow from EVM frames")
    parser.add_argument("--evm_dir", type=str, default="/home/sanghyun/robotware/svi/output/evm_frames")
    parser.add_argument("--masks_dir", type=str, default="/home/sanghyun/robotware/svi/output/sam3_masks")
    parser.add_argument("--output_dir", type=str, default="/home/sanghyun/robotware/svi/output/optical_flow_evm")
    parser.add_argument("--datasets", nargs="+", default=["normal", "sick"])
    parser.add_argument("--checkpoint_interval", type=int, default=500)

    args = parser.parse_args()

    print("=" * 60)
    print("EVM Optical Flow Extraction")
    print("=" * 60)

    evm_dir = Path(args.evm_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)

    all_summaries = {}

    for dataset in args.datasets:
        dataset_evm = evm_dir / dataset
        dataset_masks = masks_dir / dataset / "mask"
        dataset_output = output_dir / dataset

        if not dataset_evm.exists():
            print(f"EVM directory not found: {dataset_evm}")
            continue
        if not dataset_masks.exists():
            print(f"Masks directory not found: {dataset_masks}")
            continue

        summary = process_dataset(
            dataset_name=dataset,
            evm_dir=dataset_evm,
            masks_dir=dataset_masks,
            output_dir=dataset_output,
            checkpoint_interval=args.checkpoint_interval,
        )
        all_summaries[dataset] = summary

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    for dataset, summary in all_summaries.items():
        if summary:
            print(f"\n{dataset.upper()} (EVM):")
            print(f"  - Processed pairs: {summary.get('processed_pairs', 0)}")
            print(f"  - Mean IDs/frame: {summary.get('id_count', {}).get('mean', 0):.1f}")
            print(f"  - Mean activity: {summary.get('activity', {}).get('mean', 0):.4f}")


if __name__ == "__main__":
    main()
