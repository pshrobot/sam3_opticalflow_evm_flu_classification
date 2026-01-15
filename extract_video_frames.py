#!/usr/bin/env python3
"""
비디오에서 프레임 추출 스크립트
- 지정된 fps로 프레임 추출
- 출력 형식: frame_XXXX_X.Xs.jpg
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
):
    """
    비디오에서 프레임 추출

    Args:
        video_path: 입력 비디오 경로
        output_dir: 출력 디렉토리
        fps: 추출할 프레임 레이트 (기본: 1fps)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    # 프레임 간격 계산
    frame_interval = int(video_fps / fps)

    print(f"Video: {video_path.name}")
    print(f"  FPS: {video_fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Output FPS: {fps}")
    print(f"  Frame interval: every {frame_interval} frames")
    print(f"  Expected output frames: ~{int(duration * fps)}")
    print(f"  Output directory: {output_dir}")

    frame_idx = 0
    saved_count = 0

    pbar = tqdm(total=total_frames, desc="Extracting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            out_name = f"frame_{saved_count:04d}_{timestamp:.1f}s.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"\nSaved {saved_count} frames to {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
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
        help="Output directory for frames"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Output FPS (default: 1.0)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Video Frame Extraction")
    print("=" * 60)

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        fps=args.fps,
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
