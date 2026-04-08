"""SAM3 model adapter (pseudo ground-truth ceiling).

SAM3 serves as the performance ceiling / pseudo ground truth.
This adapter supports either:
  1. API endpoint (if you have SAM3 hosted)
  2. Local inference via transformers (if weights available)

Configure via endpoint_url for API mode, or model_path for local mode.
"""
import io

import numpy as np
import requests

from .base import Detection, ModelAdapter


class SAM3Adapter(ModelAdapter):
    """Adapter for SAM3 — used as pseudo ground truth / performance ceiling.

    For the benchmark, SAM3 predictions define the ceiling that SN44
    is compared against. Supports API-based inference.
    """

    name = "sam3"

    def __init__(
        self,
        endpoint_url: str,
        api_key: str | None = None,
        confidence_threshold: float = 0.25,
        class_filter: list[str] | None = None,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        # Filter to only vehicle classes to match SN44 scope
        self.class_filter = class_filter or ["bus", "car", "truck", "motorcycle"]

    def predict(self, image: np.ndarray) -> list[Detection]:
        from PIL import Image

        img = Image.fromarray(image)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.endpoint_url}/predict",
            files={"image": ("frame.jpg", buf, "image/jpeg")},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        detections = []
        for pred in data.get("predictions", data.get("results", [])):
            score = float(pred.get("confidence", pred.get("score", 0)))
            if score < self.confidence_threshold:
                continue

            class_name = str(pred.get("class_name", pred.get("label", pred.get("class", ""))))
            if self.class_filter and class_name.lower() not in [c.lower() for c in self.class_filter]:
                continue

            bbox = pred.get("bbox", pred.get("box", []))
            if isinstance(bbox, dict):
                bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
            elif "x" in pred and "width" in pred:
                cx, cy = pred["x"], pred["y"]
                w, h = pred["width"], pred["height"]
                bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

            detections.append(Detection(
                bbox=[float(b) for b in bbox],
                score=score,
                class_id=int(pred.get("class_id", pred.get("category_id", 0))),
                class_name=class_name,
            ))
        return detections
