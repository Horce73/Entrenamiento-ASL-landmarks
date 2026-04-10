from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .utils import ensure_hand_landmarker


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

FINGER_GROUPS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

FINGER_COLORS_BGR = {
    "thumb": (255, 80, 80),
    "index": (80, 220, 255),
    "middle": (90, 255, 120),
    "ring": (180, 120, 255),
    "pinky": (255, 220, 80),
    "palm": (200, 200, 200),
}

LANDMARK_TO_FINGER = {0: "palm"}
for finger_name, indices in FINGER_GROUPS.items():
    for landmark_idx in indices:
        LANDMARK_TO_FINGER[landmark_idx] = finger_name


@dataclass
class LiveProbeConfig:
    model_path: str
    labels_path: str
    hand_model_path: str = ".cache/models/hand_landmarker.task"
    min_detection_confidence: float = 0.3
    top_k: int = 3
    camera_index: int = 0


class LiveMediapipeProbe:
    def __init__(self, config: LiveProbeConfig):
        if not os.path.isfile(config.model_path):
            raise FileNotFoundError(f"No existe modelo: {config.model_path}")
        if not os.path.isfile(config.labels_path):
            raise FileNotFoundError(f"No existe labels: {config.labels_path}")

        self.config = config
        ensure_hand_landmarker(config.hand_model_path)

        self.model = tf.keras.models.load_model(config.model_path)
        self.labels = self._load_labels(config.labels_path)

        base_options = mp_python.BaseOptions(model_asset_path=config.hand_model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_detection_confidence,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

        self.total_frames = 0
        self.detected_frames = 0

    @staticmethod
    def _load_labels(labels_path: str) -> list[str]:
        with open(labels_path, "r", encoding="utf-8") as file_obj:
            return [line.strip() for line in file_obj if line.strip()]

    @staticmethod
    def _canonicalize_handedness(points: np.ndarray, handedness: str) -> np.ndarray:
        if handedness.lower() == "left":
            points = points.copy()
            points[:, 0] *= -1.0
        return points

    @staticmethod
    def _normalize_landmarks(points: np.ndarray) -> np.ndarray:
        wrist = points[0].copy()
        centered = points - wrist
        norms = np.linalg.norm(centered, axis=1)
        max_norm = float(np.max(norms))
        if max_norm > 1e-6:
            centered = centered / max_norm
        return centered.flatten().astype(np.float32)

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, hand_landmarks) -> None:
        h, w = frame.shape[:2]
        for connection in HAND_CONNECTIONS:
            finger_name = LANDMARK_TO_FINGER.get(connection[1], "palm")
            color = FINGER_COLORS_BGR.get(finger_name, (255, 255, 255))
            p1 = hand_landmarks[connection[0]]
            p2 = hand_landmarks[connection[1]]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        for idx, point in enumerate(hand_landmarks):
            finger_name = LANDMARK_TO_FINGER.get(idx, "palm")
            color = FINGER_COLORS_BGR.get(finger_name, (255, 255, 255))
            x, y = int(point.x * w), int(point.y * h)
            radius = 5 if idx in {4, 8, 12, 16, 20} else 3
            cv2.circle(frame, (x, y), radius, color, -1)

    def _predict_from_landmarks(self, hand_landmarks, handedness: str) -> list[tuple[str, float]]:
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
        points = self._canonicalize_handedness(points, handedness)
        features = self._normalize_landmarks(points)

        probs = self.model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
        top_indices = np.argsort(probs)[-self.config.top_k :][::-1]
        return [(self.labels[idx], float(probs[idx])) for idx in top_indices]

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la camara")

        print("Probe MediaPipe activo")
        print("Teclas: q=salir, p=pausa")

        paused = False
        last_time = time.time()

        try:
            while True:
                if not paused:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    self.total_frames += 1

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = self.detector.detect(mp_image)

                    if result.hand_landmarks:
                        self.detected_frames += 1
                        hand_landmarks = result.hand_landmarks[0]
                        handedness = "unknown"
                        if result.handedness and result.handedness[0]:
                            handedness = result.handedness[0][0].category_name

                        self._draw_landmarks(frame, hand_landmarks)
                        predictions = self._predict_from_landmarks(hand_landmarks, handedness)

                        cv2.putText(
                            frame,
                            f"Mano: {handedness}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                        for row, (label, prob) in enumerate(predictions):
                            y = 60 + row * 30
                            color = (0, 255, 0) if row == 0 else (255, 255, 255)
                            cv2.putText(
                                frame,
                                f"{label}: {prob * 100:.1f}%",
                                (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                color,
                                2,
                            )
                    else:
                        cv2.putText(
                            frame,
                            "Sin mano detectada",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

                    current_time = time.time()
                    fps = 1.0 / max(current_time - last_time, 1e-6)
                    last_time = current_time
                    detect_ratio = 100.0 * (self.detected_frames / max(self.total_frames, 1))

                    cv2.putText(
                        frame,
                        f"FPS: {fps:.1f} | Hand detect: {detect_ratio:.1f}%",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 220, 0),
                        2,
                    )

                    cv2.imshow("ASL MediaPipe Live Probe", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused

        finally:
            cap.release()
            cv2.destroyAllWindows()
