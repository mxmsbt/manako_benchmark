"""mAP@50 computation for object detection benchmark."""
import json
import tempfile
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..models.base import Detection


def detections_to_coco_results(
    all_detections: dict[int, list[Detection]],
) -> list[dict]:
    """Convert per-image detections to COCO results format.

    Args:
        all_detections: {image_id: [Detection, ...]}

    Returns:
        List of COCO result dicts with keys: image_id, category_id, bbox, score
    """
    results = []
    for image_id, dets in all_detections.items():
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            results.append({
                "image_id": image_id,
                "category_id": det.class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # xyxy -> xywh
                "score": det.score,
            })
    return results


def compute_map50(
    gt_coco_dict: dict,
    all_detections: dict[int, list[Detection]],
) -> dict:
    """Compute mAP@50 using pycocotools.

    Args:
        gt_coco_dict: Ground truth in COCO format.
        all_detections: {image_id: [Detection, ...]} predictions.

    Returns:
        Dict with keys: mAP50, per_class (dict of class_id -> AP50),
        precision_recall (raw arrays for plotting).
    """
    results = detections_to_coco_results(all_detections)

    if not results:
        cat_ids = [c["id"] for c in gt_coco_dict.get("categories", [])]
        return {
            "mAP50": 0.0,
            "per_class": {cid: 0.0 for cid in cat_ids},
            "num_predictions": 0,
            "num_gt": len(gt_coco_dict.get("annotations", [])),
        }

    # Write to temp files for pycocotools
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_coco_dict, f)
        gt_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        dt_path = f.name

    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(dt_path)

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.iouThrs = np.array([0.5])  # only IoU=0.5
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Overall mAP@50
        map50 = float(coco_eval.stats[0])

        # Per-class AP@50
        per_class = {}
        cat_ids = coco_gt.getCatIds()
        for cat_id in cat_ids:
            coco_eval_cls = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval_cls.params.iouThrs = np.array([0.5])
            coco_eval_cls.params.catIds = [cat_id]
            coco_eval_cls.evaluate()
            coco_eval_cls.accumulate()
            coco_eval_cls.summarize()
            per_class[cat_id] = float(coco_eval_cls.stats[0])

        return {
            "mAP50": map50,
            "per_class": per_class,
            "num_predictions": len(results),
            "num_gt": len(gt_coco_dict["annotations"]),
        }
    finally:
        Path(gt_path).unlink(missing_ok=True)
        Path(dt_path).unlink(missing_ok=True)


def compute_map50_per_class(
    gt_coco_dict: dict,
    all_detections: dict[int, list[Detection]],
) -> dict[int, float]:
    """Convenience: returns only per-class AP@50 dict."""
    result = compute_map50(gt_coco_dict, all_detections)
    return result["per_class"]
