"""Benchmark runner — orchestrates evaluation of models on a dataset."""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data.dataset import BenchmarkDataset
from ..models.base import Detection, ModelAdapter
from .metrics import compute_map50


@dataclass
class ModelResult:
    """Results for a single model on the benchmark."""
    model_name: str
    mAP50: float = 0.0
    per_class_ap: dict[int, float] = field(default_factory=dict)
    num_predictions: int = 0
    num_gt: int = 0
    avg_inference_ms: float = 0.0
    total_time_s: float = 0.0
    per_image_detections: dict[int, list[dict]] = field(default_factory=dict)
    checkpoint_tag: str = ""


@dataclass
class BenchmarkResults:
    """Complete benchmark results across all models."""
    timestamp: str = ""
    dataset_info: dict = field(default_factory=dict)
    model_results: list[ModelResult] = field(default_factory=list)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": self.timestamp,
            "dataset_info": self.dataset_info,
            "model_results": [],
        }
        for mr in self.model_results:
            d = {
                "model_name": mr.model_name,
                "mAP50": mr.mAP50,
                "per_class_ap": {str(k): v for k, v in mr.per_class_ap.items()},
                "num_predictions": mr.num_predictions,
                "num_gt": mr.num_gt,
                "avg_inference_ms": mr.avg_inference_ms,
                "total_time_s": mr.total_time_s,
                "checkpoint_tag": mr.checkpoint_tag,
            }
            data["model_results"].append(d)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkResults":
        with open(path) as f:
            data = json.load(f)
        br = cls(
            timestamp=data["timestamp"],
            dataset_info=data["dataset_info"],
        )
        for mr_data in data["model_results"]:
            br.model_results.append(ModelResult(
                model_name=mr_data["model_name"],
                mAP50=mr_data["mAP50"],
                per_class_ap={int(k): v for k, v in mr_data["per_class_ap"].items()},
                num_predictions=mr_data["num_predictions"],
                num_gt=mr_data["num_gt"],
                avg_inference_ms=mr_data["avg_inference_ms"],
                total_time_s=mr_data["total_time_s"],
                checkpoint_tag=mr_data.get("checkpoint_tag", ""),
            ))
        return br


def run_benchmark(
    dataset: BenchmarkDataset,
    models: list[ModelAdapter],
    checkpoint_tag: str = "",
) -> BenchmarkResults:
    """Run the full benchmark: evaluate every model on every frame.

    Args:
        dataset: The benchmark dataset with frames and GT annotations.
        models: List of model adapters to evaluate.
        checkpoint_tag: Optional tag for temporal tracking (e.g. "epoch_10").

    Returns:
        BenchmarkResults with mAP@50 for each model.
    """
    from datetime import datetime, timezone

    gt_coco = dataset.to_coco_dict()

    results = BenchmarkResults(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_info={
            "num_images": len(dataset),
            "num_annotations": len(dataset.annotations),
            "num_categories": len(dataset.categories),
            "categories": dataset.get_class_names(),
        },
    )

    for model in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model.name}")
        print(f"{'='*60}")

        all_detections: dict[int, list[Detection]] = {}
        per_image_dets: dict[int, list[dict]] = {}
        inference_times: list[float] = []

        for idx in tqdm(range(len(dataset)), desc=model.name):
            img_info = dataset.images[idx]
            image_id = img_info["id"]
            image = dataset.get_image(idx)

            t0 = time.perf_counter()
            dets = model.predict(image)
            t1 = time.perf_counter()

            inference_times.append((t1 - t0) * 1000)
            all_detections[image_id] = dets
            per_image_dets[image_id] = [
                {"bbox": d.bbox, "score": d.score, "class_id": d.class_id, "class_name": d.class_name}
                for d in dets
            ]

        # Compute mAP@50
        metrics = compute_map50(gt_coco, all_detections)

        model_result = ModelResult(
            model_name=model.name,
            mAP50=metrics["mAP50"],
            per_class_ap=metrics["per_class"],
            num_predictions=metrics["num_predictions"],
            num_gt=metrics["num_gt"],
            avg_inference_ms=float(np.mean(inference_times)) if inference_times else 0,
            total_time_s=sum(inference_times) / 1000,
            per_image_detections=per_image_dets,
            checkpoint_tag=checkpoint_tag,
        )
        results.model_results.append(model_result)

        print(f"  mAP@50: {model_result.mAP50:.4f}")
        print(f"  Avg inference: {model_result.avg_inference_ms:.1f}ms")
        print(f"  Predictions: {model_result.num_predictions} | GT: {model_result.num_gt}")

    return results
