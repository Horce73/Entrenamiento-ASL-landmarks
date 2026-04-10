from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .config import ValidationConfig
from .extractor import LandmarkExtractor
from .utils import VALID_EXTENSIONS, ensure_hand_landmarker


@dataclass
class ValidationSummary:
    total_images: int
    usable_images: int
    no_hand_images: int
    accuracy: float
    report_path: str
    confusion_path: str


def _load_labels(labels_path: str) -> list[str]:
    with open(labels_path, "r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def _iter_labeled_images(dataset_path: str, labels: list[str], max_images_per_class: int | None):
    # Formato 1: dataset/class_name/*.jpg
    class_dirs = [
        os.path.join(dataset_path, class_name)
        for class_name in labels
        if os.path.isdir(os.path.join(dataset_path, class_name))
    ]

    if class_dirs:
        for class_name in labels:
            class_dir = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            files = [
                name for name in sorted(os.listdir(class_dir)) if name.lower().endswith(VALID_EXTENSIONS)
            ]
            if max_images_per_class is not None:
                files = files[:max_images_per_class]

            for filename in files:
                yield class_name, os.path.join(class_dir, filename)
        return

    # Formato 2: dataset/label_algo.jpg (ej: A_test.jpg, nothing_01.png)
    flat_files = [
        name for name in sorted(os.listdir(dataset_path)) if name.lower().endswith(VALID_EXTENSIONS)
    ]
    per_class_counter = {label: 0 for label in labels}
    labels_lower = {label.lower(): label for label in labels}

    for filename in flat_files:
        stem = os.path.splitext(filename)[0].lower()
        matched_label = None
        for label_lower, label in labels_lower.items():
            if stem == label_lower or stem.startswith(f"{label_lower}_"):
                matched_label = label
                break

        if matched_label is None:
            continue

        per_class_counter[matched_label] += 1
        if max_images_per_class is not None and per_class_counter[matched_label] > max_images_per_class:
            continue

        yield matched_label, os.path.join(dataset_path, filename)


def validate_model_on_dataset(config: ValidationConfig, output_dir: str = "artifacts/landmarks_validation") -> ValidationSummary:
    if not os.path.isfile(config.model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {config.model_path}")
    if not os.path.isfile(config.labels_path):
        raise FileNotFoundError(f"Labels no encontrado: {config.labels_path}")
    if not os.path.isdir(config.dataset_path):
        raise FileNotFoundError(f"Dataset no encontrado: {config.dataset_path}")

    os.makedirs(output_dir, exist_ok=True)
    ensure_hand_landmarker(config.hand_model_path)

    labels = _load_labels(config.labels_path)
    class_to_idx = {name: idx for idx, name in enumerate(labels)}
    model = tf.keras.models.load_model(config.model_path)
    extractor = LandmarkExtractor(
        model_asset_path=config.hand_model_path,
        min_detection_confidence=config.min_detection_confidence,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    total_images = 0
    no_hand_images = 0

    for class_name, image_path in _iter_labeled_images(
        config.dataset_path,
        labels,
        config.max_images_per_class,
    ):
        total_images += 1
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        sample = extractor.extract_from_bgr(frame)
        if sample is None:
            no_hand_images += 1
            continue

        probs = model.predict(np.expand_dims(sample.features, axis=0), verbose=0)[0]
        y_true.append(class_to_idx[class_name])
        y_pred.append(int(np.argmax(probs)))

    usable_images = len(y_true)
    if usable_images == 0:
        raise RuntimeError("No hubo muestras utilizables para validar")

    y_true_np = np.asarray(y_true, dtype=np.int32)
    y_pred_np = np.asarray(y_pred, dtype=np.int32)
    accuracy = float(np.mean(y_true_np == y_pred_np))

    report = classification_report(
        y_true_np,
        y_pred_np,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true_np, y_pred_np).tolist()

    report_path = os.path.join(output_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "accuracy": accuracy,
                "total_images": total_images,
                "usable_images": usable_images,
                "no_hand_images": no_hand_images,
                "classification_report": report,
            },
            file_obj,
            indent=2,
            ensure_ascii=False,
        )

    confusion_path = os.path.join(output_dir, "validation_confusion_matrix.json")
    with open(confusion_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {"labels": labels, "matrix": confusion},
            file_obj,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Validacion accuracy: {accuracy * 100:.2f}%")
    print(f"Muestras utiles: {usable_images}/{total_images}")

    return ValidationSummary(
        total_images=total_images,
        usable_images=usable_images,
        no_hand_images=no_hand_images,
        accuracy=accuracy,
        report_path=report_path,
        confusion_path=confusion_path,
    )
