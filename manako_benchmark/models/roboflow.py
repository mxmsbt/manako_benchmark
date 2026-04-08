"""Roboflow model adapter — uses Roboflow Inference API or local SDK."""
import numpy as np

from .base import Detection, ModelAdapter


class RoboflowAdapter(ModelAdapter):
    """Adapter for Roboflow-hosted detection model.

    Can work via:
    1. Roboflow API (needs api_key + model_id)
    2. Roboflow inference SDK (local, needs `pip install inference`)
    """

    name = "roboflow"

    def __init__(
        self,
        api_key: str,
        model_id: str = "vehicles-q0x2v/1",
        confidence_threshold: float = 0.25,
        use_local: bool = False,
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.use_local = use_local
        self._model = None

        if use_local:
            self._init_local()

    def _init_local(self):
        """Initialize local Roboflow inference SDK model."""
        from inference import get_model
        self._model = get_model(model_id=self.model_id, api_key=self.api_key)

    def predict(self, image: np.ndarray) -> list[Detection]:
        if self.use_local and self._model is not None:
            return self._predict_local(image)
        return self._predict_api(image)

    def _predict_local(self, image: np.ndarray) -> list[Detection]:
        """Run inference via local Roboflow SDK."""
        import cv2
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self._model.infer(bgr, confidence=self.confidence_threshold)

        detections = []
        if isinstance(results, list):
            results = results[0]

        for pred in results.predictions:
            bbox = [
                pred.x - pred.width / 2,
                pred.y - pred.height / 2,
                pred.x + pred.width / 2,
                pred.y + pred.height / 2,
            ]
            detections.append(Detection(
                bbox=bbox,
                score=pred.confidence,
                class_id=pred.class_id if hasattr(pred, "class_id") else 0,
                class_name=pred.class_name if hasattr(pred, "class_name") else str(pred.class_id),
            ))
        return detections

    def _predict_api(self, image: np.ndarray) -> list[Detection]:
        """Run inference via Roboflow hosted API."""
        import base64
        import io
        import requests
        from PIL import Image

        img = Image.fromarray(image)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = requests.post(
            f"https://detect.roboflow.com/{self.model_id}",
            params={
                "api_key": self.api_key,
                "confidence": self.confidence_threshold,
            },
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        detections = []
        for pred in data.get("predictions", []):
            cx, cy = pred["x"], pred["y"]
            w, h = pred["width"], pred["height"]
            detections.append(Detection(
                bbox=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                score=float(pred["confidence"]),
                class_id=int(pred.get("class_id", 0)),
                class_name=str(pred.get("class", "")),
            ))
        return detections
