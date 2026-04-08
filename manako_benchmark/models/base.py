"""Base model adapter interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single detection result."""
    bbox: list[float]       # [x1, y1, x2, y2] in pixels
    score: float            # confidence score [0, 1]
    class_id: int           # integer class label
    class_name: str = ""    # optional human-readable name


class ModelAdapter(ABC):
    """Abstract adapter that all models must implement."""

    name: str = "base"

    @abstractmethod
    def predict(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a single image (RGB numpy array).

        Returns list of Detection objects.
        """

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection on a batch. Default: sequential predict."""
        return [self.predict(img) for img in images]

    def warmup(self, image_shape: tuple = (640, 640, 3)) -> None:
        """Optional warmup call before benchmarking."""
        dummy = np.zeros(image_shape, dtype=np.uint8)
        self.predict(dummy)
