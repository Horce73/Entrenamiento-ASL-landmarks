from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .config import TrainingConfig
from .dataset import build_landmark_dataset
from .extractor import LandmarkExtractor
from .modeling import build_classifier, export_tflite
from .reporting import (
    save_classification_report,
    save_confusion_matrix,
    save_labels,
    save_metadata,
    save_training_plots,
)
from .utils import ensure_hand_landmarker


@dataclass
class TrainingArtifacts:
    output_dir: str
    keras_model_path: str
    tflite_model_path: str
    labels_path: str
    metadata_path: str
    report_path: str
    training_plot_path: str
    confusion_matrix_path: str
    summary_path: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _augment_train_features(X_train: np.ndarray, y_train: np.ndarray, noise_std: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if noise_std <= 0:
        return X_train, y_train

    rng = np.random.default_rng(seed)
    noisy = X_train + rng.normal(0.0, noise_std, size=X_train.shape).astype(np.float32)
    X_aug = np.concatenate([X_train, noisy], axis=0)
    y_aug = np.concatenate([y_train, y_train], axis=0)
    return X_aug, y_aug


def train_landmark_model(config: TrainingConfig) -> TrainingArtifacts:
    if not os.path.isdir(config.dataset_path):
        raise FileNotFoundError(f"Dataset no encontrado: {config.dataset_path}")

    os.makedirs(config.output_dir, exist_ok=True)
    _set_seed(config.random_seed)

    hand_model_path = ensure_hand_landmarker(".cache/models/hand_landmarker.task")
    extractor = LandmarkExtractor(
        model_asset_path=hand_model_path,
        min_detection_confidence=config.min_detection_confidence,
    )

    dataset = build_landmark_dataset(config, extractor)
    X = dataset.X
    y = dataset.y
    class_names = dataset.class_names

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=config.test_split,
        random_state=config.random_seed,
        stratify=y,
    )

    val_ratio = config.validation_split / (1.0 - config.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=config.random_seed,
        stratify=y_temp,
    )

    X_train, y_train = _augment_train_features(X_train, y_train, config.augment_noise_std, config.random_seed)

    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight = {int(idx): float(weight) for idx, weight in zip(np.unique(y_train), class_weight_values)}

    model = build_classifier(input_dim=X.shape[1], num_classes=len(class_names), learning_rate=config.learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.output_dir, "asl_landmark_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    y_pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)
    eval_result = model.evaluate(X_test, y_test, verbose=0)
    if isinstance(eval_result, (list, tuple, np.ndarray)):
        flat = np.asarray(eval_result, dtype=np.float32).flatten().tolist()
        test_loss = float(flat[0]) if len(flat) >= 1 else 0.0
        test_accuracy = float(flat[1]) if len(flat) >= 2 else 0.0
    else:
        test_loss = float(eval_result)
        test_accuracy = 0.0

    keras_model_path = os.path.join(config.output_dir, "asl_landmark_model.keras")
    tflite_model_path = os.path.join(config.output_dir, "asl_landmark_model.tflite")
    labels_path = save_labels(class_names, config.output_dir)

    metadata = {
        "input_type": "landmarks",
        "feature_size": int(X.shape[1]),
        "num_landmarks": 21,
        "coordinates": ["x", "y", "z"],
        "normalization": "wrist_centered_maxnorm_handedness_canonical",
        "class_mode": config.class_mode,
        "num_classes": len(class_names),
        "class_names": class_names,
        "splits": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "metrics": {
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
        },
    }

    metadata_path = save_metadata(metadata, config.output_dir)
    report_path = save_classification_report(y_test, y_pred_test, class_names, config.output_dir)
    confusion_matrix_path = save_confusion_matrix(y_test, y_pred_test, class_names, config.output_dir)
    training_plot_path = save_training_plots(history, config.output_dir)
    export_tflite(model, tflite_model_path, quantization=config.quantization)

    summary_path = os.path.join(config.output_dir, "training_summary.json")
    summary = {
        "config": config.__dict__,
        "metrics": metadata["metrics"],
        "class_names": class_names,
        "extraction_stats": dataset.extraction_stats,
    }
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    print(f"Accuracy en test: {test_accuracy * 100:.2f}%")
    print(f"Salida en: {config.output_dir}")

    return TrainingArtifacts(
        output_dir=config.output_dir,
        keras_model_path=keras_model_path,
        tflite_model_path=tflite_model_path,
        labels_path=labels_path,
        metadata_path=metadata_path,
        report_path=report_path,
        training_plot_path=training_plot_path,
        confusion_matrix_path=confusion_matrix_path,
        summary_path=summary_path,
    )
