"""Demo de escritura ASL en vivo para presentaciones.

Convierte letras detectadas por el modelo de landmarks en texto con una logica
simple de estabilizacion temporal para evitar saltos por frame.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf

from asl_landmarks.extractor import LandmarkExtractor
from asl_landmarks.utils import ensure_hand_landmarker


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


COMMON_MODEL_CANDIDATES = [
    "artifacts/landmarks_full/asl_landmark_model.keras",
    "artifacts/landmarks_deep/asl_landmark_model.keras",
    "artifacts/landmarks_smoke/asl_landmark_model.keras",
]

COMMON_LABELS_CANDIDATES = [
    "artifacts/landmarks_full/labels.txt",
    "artifacts/landmarks_deep/labels.txt",
    "artifacts/landmarks_smoke/labels.txt",
    "flutter_export_landmarks_ready/labels.txt",
]


def resolve_existing_path(user_path: str, candidates: list[str], kind: str) -> str:
    if os.path.isfile(user_path):
        return user_path

    # If user passed only a basename, try common artifact locations.
    if os.path.basename(user_path) == user_path:
        for candidate in candidates:
            if os.path.basename(candidate) == user_path and os.path.isfile(candidate):
                return candidate

    existing = [path for path in candidates if os.path.isfile(path)]
    suggestions = "\n".join(f"- {path}" for path in existing) if existing else "- (sin sugerencias)"
    raise FileNotFoundError(
        f"No existe {kind}: {user_path}\n"
        f"Prueba con una ruta valida. Sugerencias disponibles:\n{suggestions}"
    )


@dataclass
class WriterConfig:
    model_path: str
    labels_path: str
    hand_model_path: str
    min_detection_confidence: float
    confidence_threshold: float
    window_size: int
    min_votes: int
    release_frames: int
    camera_index: int
    max_chars: int


class SignWriterDemo:
    def __init__(self, config: WriterConfig):
        model_path = resolve_existing_path(config.model_path, COMMON_MODEL_CANDIDATES, "modelo")
        labels_path = resolve_existing_path(config.labels_path, COMMON_LABELS_CANDIDATES, "labels")

        self.config = config
        self.config.model_path = model_path
        self.config.labels_path = labels_path

        print(f"Usando modelo: {self.config.model_path}")
        print(f"Usando labels: {self.config.labels_path}")

        ensure_hand_landmarker(config.hand_model_path)

        self.model = tf.keras.models.load_model(self.config.model_path)
        self.labels = self._load_labels(self.config.labels_path)
        self.extractor = LandmarkExtractor(
            model_asset_path=config.hand_model_path,
            min_detection_confidence=config.min_detection_confidence,
        )

        self.history: deque[str | None] = deque(maxlen=config.window_size)
        self.text_buffer: list[str] = []

        self.armed = True
        self.release_counter = 0

        self.last_top_label = "-"
        self.last_top_conf = 0.0
        self.last_committed = "-"

        self.total_frames = 0
        self.detected_frames = 0

        self.neutral_labels = {"nothing"}

    @staticmethod
    def _load_labels(path: str) -> list[str]:
        with open(path, "r", encoding="utf-8") as file_obj:
            return [line.strip() for line in file_obj if line.strip()]

    def _predict_top(self, features: np.ndarray) -> tuple[str, float]:
        probs = self.model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def _stable_label(self) -> tuple[str | None, int]:
        valid = [item for item in self.history if item is not None]
        if not valid:
            return None, 0

        label, votes = Counter(valid).most_common(1)[0]
        if votes >= self.config.min_votes:
            return label, votes
        return None, votes

    def _commit_label(self, label: str) -> None:
        if label == "space":
            if not self.text_buffer or self.text_buffer[-1] != " ":
                self.text_buffer.append(" ")
        elif label == "del":
            if self.text_buffer:
                self.text_buffer.pop()
        elif label not in self.neutral_labels:
            self.text_buffer.append(label)

        if len(self.text_buffer) > self.config.max_chars:
            overflow = len(self.text_buffer) - self.config.max_chars
            self.text_buffer = self.text_buffer[overflow:]

        self.last_committed = label

    def _display_text(self) -> str:
        text = "".join(self.text_buffer).strip()
        return text if text else "(vacio)"

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, points: np.ndarray) -> None:
        h, w = frame.shape[:2]

        for i1, i2 in HAND_CONNECTIONS:
            p1 = points[i1]
            p2 = points[i2]
            x1, y1 = int(p1[0] * w), int(p1[1] * h)
            x2, y2 = int(p2[0] * w), int(p2[1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (120, 220, 255), 2)

        for idx, p in enumerate(points):
            x, y = int(p[0] * w), int(p[1] * h)
            radius = 5 if idx in {4, 8, 12, 16, 20} else 3
            cv2.circle(frame, (x, y), radius, (80, 255, 120), -1)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la camara")

        print("ASL Sign Writer activo")
        print("Teclas: q=salir, c=limpiar texto")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                self.total_frames += 1
                sample = self.extractor.extract_from_bgr(frame)

                candidate_for_history: str | None = None
                if sample is not None:
                    self.detected_frames += 1
                    if sample.raw_points is not None and sample.raw_points.shape == (21, 3):
                        self._draw_landmarks(frame, sample.raw_points)

                    label, conf = self._predict_top(sample.features)
                    self.last_top_label = label
                    self.last_top_conf = conf

                    if conf >= self.config.confidence_threshold and label not in self.neutral_labels:
                        candidate_for_history = label
                else:
                    self.last_top_label = "-"
                    self.last_top_conf = 0.0

                self.history.append(candidate_for_history)
                stable_label, votes = self._stable_label()

                if stable_label is None:
                    self.release_counter += 1
                    if self.release_counter >= self.config.release_frames:
                        self.armed = True
                else:
                    self.release_counter = 0
                    if self.armed:
                        self._commit_label(stable_label)
                        self.armed = False

                detect_ratio = 100.0 * (self.detected_frames / max(self.total_frames, 1))
                arm_status = "LISTO" if self.armed else "ESPERA"

                cv2.putText(
                    frame,
                    f"Texto: {self._display_text()}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Top1: {self.last_top_label} ({self.last_top_conf * 100:.1f}%)",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Estable: {stable_label or '-'} | votos: {votes}/{self.config.window_size}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Estado: {arm_status} | ultimo: {self.last_committed}",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Mano: {sample.handedness if sample is not None else '-'}",
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Hand detect: {detect_ratio:.1f}% | q=salir c=limpiar",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 220, 0),
                    2,
                )

                cv2.imshow("ASL Sign Writer Demo", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    self.text_buffer.clear()
                    self.last_committed = "-"

        finally:
            cap.release()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo de escritura ASL en webcam")
    parser.add_argument("--model", default="artifacts/landmarks_full/asl_landmark_model.keras")
    parser.add_argument("--labels", default="artifacts/landmarks_full/labels.txt")
    parser.add_argument("--hand_model", default=".cache/models/hand_landmarker.task")
    parser.add_argument("--camera_index", type=int, default=0)

    parser.add_argument("--min_detection_confidence", type=float, default=0.3)
    parser.add_argument("--confidence_threshold", type=float, default=0.65)

    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--min_votes", type=int, default=5)
    parser.add_argument("--release_frames", type=int, default=4)
    parser.add_argument("--max_chars", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_votes > args.window_size:
        raise ValueError("min_votes no puede ser mayor que window_size")

    config = WriterConfig(
        model_path=args.model,
        labels_path=args.labels,
        hand_model_path=args.hand_model,
        min_detection_confidence=args.min_detection_confidence,
        confidence_threshold=args.confidence_threshold,
        window_size=args.window_size,
        min_votes=args.min_votes,
        release_frames=args.release_frames,
        camera_index=args.camera_index,
        max_chars=args.max_chars,
    )

    demo = SignWriterDemo(config)
    demo.run()


if __name__ == "__main__":
    main()
