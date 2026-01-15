"""
SAM3 Mask Extraction Script
- Batch inference를 이용한 SAM3 마스크 추출
- 시각화 이미지 + ID 마스크 이미지 출력
- 겹치는 영역은 작은 박스의 값 우선
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import logging
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_BPE_PATH = "/home/sanghyun/robotware/svi/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"


def get_color_map(num_colors=256):
    """Generate a color map for visualization."""
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_colors, 3), dtype=np.uint8)
    return colors


def compute_box_area(box: List[float]) -> float:
    """Compute box area from [x1, y1, x2, y2] format."""
    return (box[2] - box[0]) * (box[3] - box[1])


def create_id_mask(masks: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor,
                   height: int, width: int) -> np.ndarray:
    """
    Create ID mask where each pixel has the object ID value.
    Overlapping regions use the smaller box's ID.
    """
    if len(masks) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    masks_np = masks.squeeze(1).cpu().numpy()
    boxes_np = boxes.cpu().numpy()

    # Sort by area (descending - larger first, so smaller boxes overwrite)
    areas = [(compute_box_area(box), idx) for idx, box in enumerate(boxes_np)]
    areas.sort(reverse=True, key=lambda x: x[0])

    id_mask = np.zeros((height, width), dtype=np.uint8)

    for area, orig_idx in areas:
        mask = masks_np[orig_idx] > 0
        id_mask[mask] = orig_idx + 1

    return id_mask


def create_visualization(frame: np.ndarray, masks: torch.Tensor, boxes: torch.Tensor,
                         scores: torch.Tensor, colors: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create visualization with overlaid masks."""
    if len(masks) == 0:
        return frame.copy()

    overlay = frame.copy()
    masks_np = masks.squeeze(1).cpu().numpy()
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()

    for idx in range(len(masks_np)):
        mask = masks_np[idx] > 0
        color = colors[idx % len(colors)]

        overlay[mask] = (
            overlay[mask] * (1 - alpha) +
            np.array(color, dtype=np.float32) * alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

        box = boxes_np[idx].astype(int)
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color.tolist(), 1)

        score = scores_np[idx]
        label = f"ID:{idx+1} ({score:.2f})"
        cv2.putText(overlay, label, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(overlay, label, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color.tolist(), 1)

    info = f"Objects: {len(masks_np)}"
    cv2.putText(overlay, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(overlay, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return overlay


class FrameDataset(Dataset):
    """Dataset for loading frames."""

    def __init__(self, frame_paths: List[Path]):
        self.frame_paths = frame_paths

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        frame_bgr = cv2.imread(str(frame_path))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return {
            'pil_image': pil_image,
            'frame_bgr': frame_bgr,
            'frame_path': str(frame_path),
            'frame_name': frame_path.stem,
        }


class SAM3BatchProcessor:
    """SAM3 배치 프로세서 - backbone 배치 처리"""

    def __init__(self, text_prompt: str = "chicken", threshold: float = 0.1,
                 device: str = "cuda", batch_size: int = 8):
        self.text_prompt = text_prompt
        self.threshold = threshold
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load SAM3 model"""
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        logger.info("Loading SAM3 model...")
        self.model = build_sam3_image_model(
            bpe_path=DEFAULT_BPE_PATH,
            device=self.device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=1008,
            device=self.device,
            confidence_threshold=self.threshold,
        )

        # Pre-compute text features (can be reused)
        self._precompute_text_features()
        logger.info("SAM3 model loaded successfully!")

    def _precompute_text_features(self):
        """Pre-compute text features for the prompt."""
        self.text_outputs = self.model.backbone.forward_text(
            [self.text_prompt], device=self.device
        )

    @torch.inference_mode()
    def process_batch(self, pil_images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Process a batch of images with batch backbone forward.

        Args:
            pil_images: List of PIL images

        Returns:
            List of result dicts with masks, boxes, scores
        """
        from torchvision.transforms import v2
        from sam3.model.data_misc import interpolate
        from sam3.model import box_ops

        if not pil_images:
            return []

        # Get original dimensions
        original_sizes = [(img.height, img.width) for img in pil_images]

        # Transform images for batch processing
        transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(self.processor.resolution, self.processor.resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Batch transform
        batch_tensors = []
        for img in pil_images:
            img_tensor = v2.functional.to_image(img).to(self.device)
            img_tensor = transform(img_tensor)
            batch_tensors.append(img_tensor)

        batch_input = torch.stack(batch_tensors, dim=0)  # (B, 3, H, W)

        # Batch backbone forward
        backbone_out = self.model.backbone.forward_image(batch_input)

        # Process each image's features with text prompt
        results = []
        batch_size = len(pil_images)

        for i in range(batch_size):
            # Extract single image's backbone features
            single_backbone_out = self._extract_single_backbone(backbone_out, i)
            single_backbone_out.update(self.text_outputs)

            # Get geometric prompt
            geometric_prompt = self.model._get_dummy_prompt()

            # Forward grounding for single image
            from sam3.model.data_misc import FindStage
            find_stage = FindStage(
                img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

            outputs = self.model.forward_grounding(
                backbone_out=single_backbone_out,
                find_input=find_stage,
                geometric_prompt=geometric_prompt,
                find_target=None,
            )

            # Process outputs
            out_bbox = outputs["pred_boxes"]
            out_logits = outputs["pred_logits"]
            out_masks = outputs["pred_masks"]
            out_probs = out_logits.sigmoid()
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = (out_probs * presence_score).squeeze(-1)

            keep = out_probs > self.threshold
            out_probs = out_probs[keep]
            out_masks = out_masks[keep]
            out_bbox = out_bbox[keep]

            if len(out_probs) > 0:
                boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
                img_h, img_w = original_sizes[i]
                scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
                boxes = boxes * scale_fct[None, :]

                out_masks = interpolate(
                    out_masks.unsqueeze(1),
                    (img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()

                masks = out_masks > 0.5
            else:
                masks = torch.tensor([])
                boxes = torch.tensor([])
                out_probs = torch.tensor([])

            results.append({
                'masks': masks,
                'boxes': boxes,
                'scores': out_probs,
                'height': original_sizes[i][0],
                'width': original_sizes[i][1],
            })

        return results

    def _extract_single_backbone(self, backbone_out: Dict, idx: int) -> Dict:
        """Extract single image's backbone features from batch output."""
        single_out = {}

        for key, value in backbone_out.items():
            if value is None:
                single_out[key] = None
            elif key == 'sam2_backbone_out':
                # Handle nested sam2 backbone output
                single_sam2 = {}
                for k2, v2 in value.items():
                    if v2 is None:
                        single_sam2[k2] = None
                    elif k2 == 'backbone_fpn':
                        single_sam2[k2] = [feat[idx:idx+1] for feat in v2]
                    elif k2 == 'vision_pos_enc':
                        single_sam2[k2] = [pos[idx:idx+1] for pos in v2]
                    elif isinstance(v2, torch.Tensor):
                        single_sam2[k2] = v2[idx:idx+1]
                    else:
                        single_sam2[k2] = v2
                single_out[key] = single_sam2
            elif isinstance(value, torch.Tensor):
                single_out[key] = value[idx:idx+1]
            elif isinstance(value, list):
                single_out[key] = [v[idx:idx+1] if isinstance(v, torch.Tensor) else v for v in value]
            else:
                single_out[key] = value

        return single_out


def process_frames_directory(
    frames_dir: Path,
    output_dir: Path,
    processor: SAM3BatchProcessor,
    colors: np.ndarray,
    max_frames: int = None,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """Process all frames in a directory with batch inference."""

    vis_dir = output_dir / "visualization"
    mask_dir = output_dir / "mask"
    vis_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob('*.jpg'))
    if max_frames:
        frame_files = frame_files[:max_frames]

    # Skip already processed frames
    existing_masks = set(f.stem for f in mask_dir.glob('*.png'))
    original_count = len(frame_files)
    frame_files = [f for f in frame_files if f.stem not in existing_masks]
    skipped_count = original_count - len(frame_files)

    if skipped_count > 0:
        logger.info(f"Skipping {skipped_count} already processed frames")

    logger.info(f"Processing {len(frame_files)} frames from {frames_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Output visualization: {vis_dir}")
    logger.info(f"Output masks: {mask_dir}")

    total_detections = 0
    frames_with_detections = 0

    # Process in batches
    pbar = tqdm(total=len(frame_files), desc=f"Processing {frames_dir.parent.name}")

    for batch_start in range(0, len(frame_files), batch_size):
        batch_end = min(batch_start + batch_size, len(frame_files))
        batch_paths = frame_files[batch_start:batch_end]

        # Load batch
        pil_images = []
        frames_bgr = []
        frame_names = []

        for frame_path in batch_paths:
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            pil_images.append(pil_image)
            frames_bgr.append(frame_bgr)
            frame_names.append(frame_path.stem)

        if not pil_images:
            continue

        # Batch inference
        results = processor.process_batch(pil_images)

        # Save results
        for frame_bgr, frame_name, result in zip(frames_bgr, frame_names, results):
            masks = result['masks']
            boxes = result['boxes']
            scores = result['scores']
            height = result['height']
            width = result['width']

            n_objs = len(masks) if hasattr(masks, '__len__') and len(masks.shape) > 0 else 0

            if n_objs > 0:
                total_detections += n_objs
                frames_with_detections += 1
                vis_frame = create_visualization(frame_bgr, masks, boxes, scores, colors)
                id_mask = create_id_mask(masks, boxes, scores, height, width)
            else:
                vis_frame = frame_bgr.copy()
                cv2.putText(vis_frame, "No detections", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                id_mask = np.zeros((height, width), dtype=np.uint8)

            vis_path = vis_dir / f"{frame_name}.jpg"
            mask_path = mask_dir / f"{frame_name}.png"

            cv2.imwrite(str(vis_path), vis_frame)
            cv2.imwrite(str(mask_path), id_mask)

        pbar.update(len(batch_paths))
        pbar.set_postfix({
            'det': total_detections,
            'frames_w_det': frames_with_detections
        })

    pbar.close()

    logger.info(f"Completed: {len(frame_files)} frames")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Frames with detections: {frames_with_detections}")

    return {
        'total_frames': len(frame_files),
        'total_detections': total_detections,
        'frames_with_detections': frames_with_detections,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Mask Extraction with Batch Inference")
    parser.add_argument("--data_dir", type=str,
                        default="/home/sanghyun/robotware/svi/data")
    parser.add_argument("--output_dir", type=str,
                        default="/home/sanghyun/robotware/svi/output/sam3_masks")
    parser.add_argument("--text_prompt", type=str, default="chicken")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--datasets", nargs="+", default=["normal", "sick"])

    args = parser.parse_args()

    print("=" * 60)
    print("SAM3 Mask Extraction - Batch Inference")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Threshold: {args.threshold}")
    print(f"Text prompt: {args.text_prompt}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets: {args.datasets}")
    print(f"Max frames: {args.max_frames or 'all'}")
    print("=" * 60)

    processor = SAM3BatchProcessor(
        text_prompt=args.text_prompt,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
    )

    colors = get_color_map(256)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    all_stats = {}

    for dataset in args.datasets:
        frames_dir = data_dir / dataset / "frames"

        if not frames_dir.exists():
            logger.warning(f"Frames directory not found: {frames_dir}")
            continue

        dataset_output_dir = output_dir / dataset

        print(f"\n{'=' * 60}")
        print(f"Processing: {dataset}")
        print(f"{'=' * 60}")

        stats = process_frames_directory(
            frames_dir=frames_dir,
            output_dir=dataset_output_dir,
            processor=processor,
            colors=colors,
            max_frames=args.max_frames,
            batch_size=args.batch_size,
        )

        all_stats[dataset] = stats

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    for dataset, stats in all_stats.items():
        print(f"\n{dataset.upper()}:")
        print(f"  - Total frames: {stats['total_frames']}")
        print(f"  - Total detections: {stats['total_detections']}")
        print(f"  - Frames with detections: {stats['frames_with_detections']}")
        print(f"  - Output: {output_dir / dataset}")
    print("=" * 60)


if __name__ == "__main__":
    main()
