"""Dataset loader for custom evaluation frames with COCO-format annotations."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


class BenchmarkDataset:
    """Loads evaluation frames and their ground-truth annotations.

    Supports two annotation formats:
    1. COCO JSON: a single annotations.json with COCO object detection format
    2. Per-frame YOLO-style: one .txt file per image with class_id cx cy w h (normalized)

    Directory structure expected:
        data/
          frames/          # .jpg, .png, .webp images
          annotations/     # annotations.json (COCO) OR per-frame .txt files
    """

    def __init__(self, frames_dir: str | Path, annotations_path: str | Path):
        self.frames_dir = Path(frames_dir)
        self.annotations_path = Path(annotations_path)
        self.images: list[dict] = []        # [{id, file_name, width, height}]
        self.annotations: list[dict] = []   # [{id, image_id, category_id, bbox:[x,y,w,h]}]
        self.categories: list[dict] = []    # [{id, name}]
        self._category_map: dict[int, str] = {}

        self._load()

    def _load(self):
        if self.annotations_path.suffix == ".json":
            self._load_coco()
        elif self.annotations_path.is_dir():
            self._load_yolo(self.annotations_path)
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotations_path}")

    def _load_coco(self):
        with open(self.annotations_path) as f:
            coco = json.load(f)

        self.images = coco.get("images", [])
        self.annotations = coco.get("annotations", [])
        self.categories = coco.get("categories", [])
        self._category_map = {c["id"]: c["name"] for c in self.categories}

        # Validate frames exist
        missing = [img for img in self.images if not (self.frames_dir / img["file_name"]).exists()]
        if missing:
            names = [m["file_name"] for m in missing[:5]]
            raise FileNotFoundError(f"Missing {len(missing)} frames, e.g.: {names}")

    def _load_yolo(self, labels_dir: Path):
        """Load YOLO-format labels: class_id center_x center_y width height (normalized)."""
        frame_files = sorted(
            p for p in self.frames_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )

        ann_id = 0
        for img_id, frame_path in enumerate(frame_files):
            img = Image.open(frame_path)
            w, h = img.size
            self.images.append({
                "id": img_id,
                "file_name": frame_path.name,
                "width": w,
                "height": h,
            })

            label_file = labels_dir / f"{frame_path.stem}.txt"
            if not label_file.exists():
                continue

            for line in label_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Convert normalized center-format to COCO [x, y, w, h] in pixels
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                box_w = bw * w
                box_h = bh * h
                self.annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                })
                ann_id += 1

                if cls_id not in self._category_map:
                    self._category_map[cls_id] = str(cls_id)
                    self.categories.append({"id": cls_id, "name": str(cls_id)})

    def __len__(self) -> int:
        return len(self.images)

    def get_image(self, idx: int) -> np.ndarray:
        """Load image as RGB numpy array."""
        img_info = self.images[idx]
        path = self.frames_dir / img_info["file_name"]
        return np.array(Image.open(path).convert("RGB"))

    def get_gt_boxes(self, image_id: int) -> list[dict]:
        """Get ground-truth annotations for an image.

        Returns list of dicts with keys: bbox (xyxy), category_id, class_name.
        """
        results = []
        for ann in self.annotations:
            if ann["image_id"] != image_id:
                continue
            x, y, w, h = ann["bbox"]
            results.append({
                "bbox": [x, y, x + w, y + h],  # convert to xyxy
                "category_id": ann["category_id"],
                "class_name": self._category_map.get(ann["category_id"], ""),
            })
        return results

    def get_class_names(self) -> dict[int, str]:
        return dict(self._category_map)

    def to_coco_dict(self) -> dict:
        """Export as COCO-format dict (for pycocotools compatibility)."""
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }
