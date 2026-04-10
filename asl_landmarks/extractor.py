from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


@dataclass
class LandmarkSample:
    features: np.ndarray
    handedness: str
    raw_points: Optional[np.ndarray] = None


class LandmarkExtractor:
    def __init__(self, model_asset_path: str, min_detection_confidence: float = 0.3):
        base_options = mp_python.BaseOptions(model_asset_path=model_asset_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _normalize_landmarks(points: np.ndarray) -> np.ndarray:
        wrist = points[0].copy()
        centered = points - wrist
        norms = np.linalg.norm(centered, axis=1)
        max_norm = float(np.max(norms))
        if max_norm > 1e-6:
            centered = centered / max_norm
        return centered.astype(np.float32)

    @staticmethod
    def _canonicalize_handedness(points: np.ndarray, handedness: str) -> np.ndarray:
        if handedness.lower() == "left":
            points = points.copy()
            points[:, 0] *= -1.0
        return points

    def extract_from_bgr(self, image_bgr: np.ndarray) -> Optional[LandmarkSample]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)
        if not result.hand_landmarks:
            return None

        points = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=np.float32)
        raw_points = points.copy()
        handedness = "unknown"
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name

        points = self._canonicalize_handedness(points, handedness)
        normalized = self._normalize_landmarks(points)
        return LandmarkSample(features=normalized.flatten(), handedness=handedness, raw_points=raw_points)

    def extract_from_path(self, image_path: str) -> Optional[LandmarkSample]:
        image = cv2.imread(image_path)
        if image is None:
            return None
        return self.extract_from_bgr(image)
