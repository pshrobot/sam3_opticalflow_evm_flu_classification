#!/usr/bin/env python3
"""
Eulerian Video Magnification (EVM) 적용 스크립트

원본 비디오에서 미세한 움직임을 증폭하여 프레임으로 저장합니다.
기존 1fps 프레임과 동일한 타임스탬프로 샘플링합니다.
"""

import os
import cv2
import numpy as np
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from eulerian_magnification import eulerian_magnification

warnings.filterwarnings('ignore')


def get_pyramid_safe_size(width, height, pyramid_levels=4):
    """피라미드 처리에 안전한 크기로 조정"""
    divisor = 2 ** pyramid_levels
    new_width = (width // divisor) * divisor
    new_height = (height // divisor) * divisor
    return new_width, new_height


def process_video_chunk(
    frames: np.ndarray,
    fps: float,
    freq_min: float = 0.5,
    freq_max: float = 2.0,
    amplification: float = 20,
    pyramid_levels: int = 4
) -> np.ndarray:
    """
    프레임 청크에 EVM 적용

    Args:
        frames: (N, H, W, C) float32 배열 [0, 1]
        fps: 프레임 레이트
        freq_min: 저주파 컷오프 (Hz)
        freq_max: 고주파 컷오프 (Hz)
        amplification: 증폭 계수
        pyramid_levels: 피라미드 레벨 수

    Returns:
        증폭된 프레임 배열
    """
    magnified = eulerian_magnification(
        frames,
        fps=fps,
        freq_min=freq_min,
        freq_max=freq_max,
        amplification=amplification,
        pyramid_levels=pyramid_levels
    )
    return np.clip(magnified, 0, 1)


def apply_evm_to_video(
    video_path: Path,
    output_dir: Path,
    chunk_size: int = 300,  # 30초 분량 (10fps 기준)
    overlap: int = 50,      # 청크 간 오버랩
    resize_factor: float = 0.5,
    freq_min: float = 0.5,
    freq_max: float = 2.0,
    amplification: float = 20,
    pyramid_levels: int = 4,
    sample_interval: int = 10,  # 10프레임마다 1개 저장 (1fps)
):
    """
    비디오에 EVM을 적용하고 프레임으로 저장

    Args:
        video_path: 입력 비디오 경로
        output_dir: 출력 디렉토리
        chunk_size: 한 번에 처리할 프레임 수
        overlap: 청크 간 오버랩 프레임 수
        resize_factor: 리사이즈 비율 (처리 속도용)
        freq_min, freq_max: 주파수 범위
        amplification: 증폭 계수
        pyramid_levels: 피라미드 레벨 수
        sample_interval: 저장할 프레임 간격
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 비디오 정보 읽기
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 리사이즈 크기 계산 (피라미드 안전 크기)
    new_width = int(orig_width * resize_factor)
    new_height = int(orig_height * resize_factor)
    safe_width, safe_height = get_pyramid_safe_size(new_width, new_height, pyramid_levels)

    print(f"Video: {video_path.name}")
    print(f"  Original: {orig_width}x{orig_height} @ {fps}fps")
    print(f"  Processing: {safe_width}x{safe_height}")
    print(f"  Total frames: {total_frames}")
    print(f"  EVM params: freq={freq_min}-{freq_max}Hz, amp={amplification}")
    print(f"  Sample interval: every {sample_interval} frames ({fps/sample_interval:.1f} fps output)")

    # 출력 프레임 수 예상
    expected_output = total_frames // sample_interval
    print(f"  Expected output frames: ~{expected_output}")

    # 처리 시작
    frame_idx = 0
    saved_count = 0
    step = chunk_size - overlap

    pbar = tqdm(total=total_frames, desc="Processing")

    while True:
        # 청크 읽기
        frames = []
        chunk_start = frame_idx

        for i in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break

            # 리사이즈 및 정규화
            frame_resized = cv2.resize(frame, (safe_width, safe_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.float32) / 255.0)
            frame_idx += 1
            pbar.update(1)

        if len(frames) < 30:  # 최소 3초 분량 필요
            break

        # EVM 적용
        vid_data = np.array(frames)
        try:
            magnified = process_video_chunk(
                vid_data, fps, freq_min, freq_max, amplification, pyramid_levels
            )
        except Exception as e:
            print(f"\nWarning: EVM failed for chunk at frame {chunk_start}: {e}")
            continue

        # 프레임 저장 (오버랩 부분은 제외)
        save_start = overlap if chunk_start > 0 else 0
        save_end = len(magnified) if frame_idx >= total_frames else len(magnified) - overlap

        for i in range(save_start, save_end):
            global_idx = chunk_start + i

            # sample_interval 간격으로만 저장
            if global_idx % sample_interval == 0:
                # 타임스탬프 계산
                timestamp = global_idx / fps

                # 프레임 저장
                frame_out = (magnified[i] * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)

                # 원본 크기로 복원
                frame_full = cv2.resize(frame_bgr, (orig_width, orig_height))

                out_name = f"frame_{timestamp:.1f}s.jpg"
                cv2.imwrite(str(output_dir / out_name), frame_full)
                saved_count += 1

        # 다음 청크 위치로 이동
        if frame_idx >= total_frames:
            break

        # 오버랩 위치로 되돌리기
        cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start + step)
        frame_idx = chunk_start + step

    pbar.close()
    cap.release()

    print(f"\nSaved {saved_count} frames to {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Apply Eulerian Video Magnification")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for magnified frames"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=300,
        help="Frames per chunk (default: 300, ~30s at 10fps)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks (default: 50)"
    )
    parser.add_argument(
        "--resize_factor",
        type=float,
        default=0.5,
        help="Resize factor for processing (default: 0.5)"
    )
    parser.add_argument(
        "--freq_min",
        type=float,
        default=0.5,
        help="Minimum frequency in Hz (default: 0.5)"
    )
    parser.add_argument(
        "--freq_max",
        type=float,
        default=2.0,
        help="Maximum frequency in Hz (default: 2.0)"
    )
    parser.add_argument(
        "--amplification",
        type=float,
        default=20,
        help="Amplification factor (default: 20)"
    )
    parser.add_argument(
        "--pyramid_levels",
        type=int,
        default=4,
        help="Pyramid levels (default: 4)"
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10,
        help="Save every N frames (default: 10 for 1fps output)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Eulerian Video Magnification")
    print("=" * 60)

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    start_time = datetime.now()

    apply_evm_to_video(
        video_path=video_path,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        resize_factor=args.resize_factor,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        amplification=args.amplification,
        pyramid_levels=args.pyramid_levels,
        sample_interval=args.sample_interval,
    )

    elapsed = datetime.now() - start_time
    print(f"\nTotal time: {elapsed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
