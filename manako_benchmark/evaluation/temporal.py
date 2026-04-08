"""Temporal tracking: track model improvement across checkpoints."""
import json
from pathlib import Path

from .runner import BenchmarkResults


class TemporalTracker:
    """Tracks benchmark results over time / across checkpoints.

    Stores a history of BenchmarkResults, keyed by checkpoint tag,
    to show how models improve through subnet training.
    """

    def __init__(self, history_path: str | Path):
        self.history_path = Path(history_path)
        self.history: list[dict] = []
        if self.history_path.exists():
            with open(self.history_path) as f:
                self.history = json.load(f)

    def add_result(self, results: BenchmarkResults):
        """Append a benchmark run to the history."""
        entry = {
            "timestamp": results.timestamp,
            "dataset_info": results.dataset_info,
            "models": {},
        }
        for mr in results.model_results:
            entry["models"][mr.model_name] = {
                "mAP50": mr.mAP50,
                "per_class_ap": {str(k): v for k, v in mr.per_class_ap.items()},
                "avg_inference_ms": mr.avg_inference_ms,
                "checkpoint_tag": mr.checkpoint_tag,
            }
        self.history.append(entry)
        self._save()

    def _save(self):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_model_timeline(self, model_name: str) -> list[dict]:
        """Get mAP@50 progression for a specific model across checkpoints."""
        timeline = []
        for entry in self.history:
            model_data = entry["models"].get(model_name)
            if model_data:
                timeline.append({
                    "timestamp": entry["timestamp"],
                    "checkpoint_tag": model_data.get("checkpoint_tag", ""),
                    "mAP50": model_data["mAP50"],
                    "avg_inference_ms": model_data["avg_inference_ms"],
                })
        return timeline

    def get_comparison_table(self) -> list[dict]:
        """Get latest results for all models as a comparison."""
        if not self.history:
            return []
        latest = self.history[-1]
        rows = []
        for model_name, data in latest["models"].items():
            rows.append({
                "model": model_name,
                "mAP50": data["mAP50"],
                "avg_inference_ms": data["avg_inference_ms"],
            })
        return sorted(rows, key=lambda r: r["mAP50"], reverse=True)

    def get_improvement_summary(self, model_name: str) -> dict | None:
        """Get improvement from first to latest checkpoint for a model."""
        timeline = self.get_model_timeline(model_name)
        if len(timeline) < 2:
            return None
        first, last = timeline[0], timeline[-1]
        return {
            "model": model_name,
            "initial_mAP50": first["mAP50"],
            "final_mAP50": last["mAP50"],
            "absolute_improvement": last["mAP50"] - first["mAP50"],
            "relative_improvement_pct": (
                ((last["mAP50"] - first["mAP50"]) / first["mAP50"] * 100)
                if first["mAP50"] > 0 else float("inf")
            ),
            "num_checkpoints": len(timeline),
            "first_checkpoint": first.get("checkpoint_tag", ""),
            "last_checkpoint": last.get("checkpoint_tag", ""),
        }
