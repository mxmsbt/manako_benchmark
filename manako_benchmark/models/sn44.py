"""SN44 (Manako subnet-trained) model adapter — local ONNX inference."""
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from pathlib import Path

from .base import Detection, ModelAdapter

CLASS_NAMES = ["bus", "car", "truck", "motorcycle"]


class SN44Adapter(ModelAdapter):
    """Adapter for SN44 subnet-trained YOLO detection model (ONNX).

    Model: alfred8995/kane001 on HuggingFace
    Input: 960x960 letterboxed, RGB, float32 [0,1]
    Output: [N, 6] = [x1, y1, x2, y2, conf, cls_id]
    """

    name = "sn44"

    def __init__(
        self,
        model_path: str | Path | None = None,
        hf_repo: str = "alfred8995/kane001",
        hf_filename: str = "weights.onnx",
        input_size: int = 960,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Download from HuggingFace if no local path
        if model_path is None:
            model_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
        self.model_path = str(model_path)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Resize with aspect ratio preservation + padding to input_size x input_size."""
        h, w = image.shape[:2]
        target = self.input_size
        ratio = min(target / w, target / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (target - new_w) / 2
        pad_h = (target - new_h) / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return padded, ratio, (pad_w, pad_h)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Preprocess: letterbox, BGR->RGB, normalize, HWC->NCHW."""
        # Input is RGB from PIL, convert to BGR for letterbox then back
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        padded, ratio, pad = self._letterbox(bgr)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        tensor = np.ascontiguousarray(tensor)
        return tensor, ratio, pad

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        """Hard NMS."""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def _postprocess(
        self, output: np.ndarray, ratio: float, pad: tuple[float, float], orig_size: tuple[int, int]
    ) -> list[Detection]:
        """Decode ONNX output to Detection list."""
        output = np.squeeze(output)

        # Handle different output formats
        if output.ndim == 2:
            if output.shape[1] == 6:
                # [N, 6] = [x1, y1, x2, y2, conf, cls_id]
                boxes = output[:, :4]
                scores = output[:, 4]
                class_ids = output[:, 5].astype(int)
            elif output.shape[0] == 6 or output.shape[0] == (4 + len(CLASS_NAMES)):
                # [C, N] format — transpose
                output = output.T
                boxes = output[:, :4]
                if output.shape[1] == 6:
                    scores = output[:, 4]
                    class_ids = output[:, 5].astype(int)
                else:
                    class_probs = output[:, 4:]
                    class_ids = class_probs.argmax(axis=1)
                    scores = class_probs[np.arange(len(class_probs)), class_ids]
            else:
                return []
        else:
            return []

        # Confidence filter
        mask = scores >= self.conf_threshold
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return []

        # Reverse letterbox: remove padding, then scale to original
        pad_w, pad_h = pad
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= ratio

        # Clip to image bounds
        orig_w, orig_h = orig_size
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # Filter degenerate boxes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths > 2) & (heights > 2) & (widths * heights >= 196)
        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]

        # Per-class NMS
        detections = []
        for cls_id in np.unique(class_ids):
            cls_mask = class_ids == cls_id
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            keep = self._nms(cls_boxes, cls_scores)
            for idx in keep:
                detections.append(Detection(
                    bbox=cls_boxes[idx].tolist(),
                    score=float(cls_scores[idx]),
                    class_id=int(cls_id),
                    class_name=CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else str(cls_id),
                ))

        return sorted(detections, key=lambda d: d.score, reverse=True)

    def predict(self, image: np.ndarray) -> list[Detection]:
        orig_h, orig_w = image.shape[:2]
        tensor, ratio, pad = self._preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self._postprocess(outputs[0], ratio, pad, (orig_w, orig_h))
