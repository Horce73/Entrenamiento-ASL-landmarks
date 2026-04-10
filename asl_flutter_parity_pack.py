"""Generate a Flutter parity pack for ASL landmark TFLite integration.

This script extracts landmarks from labeled images using the same Python pipeline,
runs inference with the TFLite model, and exports JSON cases that can be replayed
in Flutter to verify input/output parity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from asl_landmarks.extractor import LandmarkExtractor
from asl_landmarks.utils import VALID_EXTENSIONS, ensure_hand_landmarker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build parity cases for Flutter vs Python")
    parser.add_argument("--model_tflite", default="artifacts/landmarks_full/asl_landmark_model.tflite")
    parser.add_argument("--labels", default="artifacts/landmarks_full/labels.txt")
    parser.add_argument("--dataset", default="archive/asl_alphabet_test/asl_alphabet_test")
    parser.add_argument("--output", default="artifacts/flutter_parity/parity_pack.json")
    parser.add_argument("--max_images_per_class", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--hand_model", default=".cache/models/hand_landmarker.task")
    parser.add_argument("--min_detection_confidence", type=float, default=0.3)
    return parser.parse_args()


def load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def iter_labeled_images(
    dataset_path: str,
    labels: list[str],
    max_images_per_class: int | None,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    # Format A: dataset/class_name/*.jpg
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
                pairs.append((class_name, os.path.join(class_dir, filename)))
        return pairs

    # Format B: dataset/label_xxx.jpg (e.g. A_test.jpg, space_01.jpg)
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

        pairs.append((matched_label, os.path.join(dataset_path, filename)))

    return pairs


def build_tflite(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details


def run_tflite(
    interpreter: tf.lite.Interpreter,
    input_details: dict,
    output_details: dict,
    features: np.ndarray,
) -> np.ndarray:
    input_shape = [int(v) for v in input_details["shape"].tolist()]
    input_dtype = input_details["dtype"]

    x = np.asarray(features, dtype=input_dtype).reshape(input_shape)
    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details["index"])[0]
    return np.asarray(probs, dtype=np.float32)


def to_rel_path(path: str) -> str:
    try:
        return os.path.relpath(path, os.getcwd())
    except ValueError:
        return path


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.model_tflite):
        raise FileNotFoundError(f"Model not found: {args.model_tflite}")
    if not os.path.isfile(args.labels):
        raise FileNotFoundError(f"Labels not found: {args.labels}")
    if not os.path.isdir(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    ensure_hand_landmarker(args.hand_model)

    labels = load_labels(args.labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    extractor = LandmarkExtractor(
        model_asset_path=args.hand_model,
        min_detection_confidence=args.min_detection_confidence,
    )

    interpreter, input_details, output_details = build_tflite(args.model_tflite)

    image_pairs = iter_labeled_images(args.dataset, labels, args.max_images_per_class)
    if not image_pairs:
        raise RuntimeError("No labeled images found for parity pack generation")

    total_images = 0
    usable_images = 0
    no_hand_images = 0
    read_errors = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    cases: list[dict] = []
    no_hand_cases: list[dict] = []

    top_k = max(1, int(args.top_k))

    for expected_label, image_path in image_pairs:
        total_images += 1
        frame = cv2.imread(image_path)
        if frame is None:
            read_errors += 1
            no_hand_cases.append(
                {
                    "image_path": to_rel_path(image_path),
                    "expected_label": expected_label,
                    "reason": "read_error",
                }
            )
            continue

        sample = extractor.extract_from_bgr(frame)
        if sample is None:
            no_hand_images += 1
            no_hand_cases.append(
                {
                    "image_path": to_rel_path(image_path),
                    "expected_label": expected_label,
                    "reason": "no_hand_detected",
                }
            )
            continue

        probs = run_tflite(interpreter, input_details, output_details, sample.features)
        pred_idx = int(np.argmax(probs))
        top_indices = np.argsort(probs)[-top_k:][::-1]

        expected_idx = label_to_idx.get(expected_label)
        if expected_idx is not None:
            y_true.append(int(expected_idx))
            y_pred.append(pred_idx)

        usable_images += 1
        cases.append(
            {
                "id": len(cases),
                "image_path": to_rel_path(image_path),
                "expected_label": expected_label,
                "expected_index": int(expected_idx) if expected_idx is not None else None,
                "predicted_label": labels[pred_idx] if 0 <= pred_idx < len(labels) else str(pred_idx),
                "predicted_index": pred_idx,
                "confidence": float(probs[pred_idx]),
                "handedness": sample.handedness,
                "features_63": [float(x) for x in sample.features.tolist()],
                "top_k": [
                    {
                        "index": int(idx),
                        "label": labels[int(idx)] if 0 <= int(idx) < len(labels) else str(int(idx)),
                        "prob": float(probs[int(idx)]),
                    }
                    for idx in top_indices
                ],
            }
        )

    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else 0.0

    input_shape = [int(v) for v in input_details["shape"].tolist()]
    output_shape = [int(v) for v in output_details["shape"].tolist()]

    parity_pack = {
        "summary": {
            "total_images": total_images,
            "usable_images": usable_images,
            "no_hand_images": no_hand_images,
            "read_errors": read_errors,
            "top1_accuracy": accuracy,
        },
        "contract": {
            "input_shape": input_shape,
            "input_dtype": str(np.dtype(input_details["dtype"])),
            "output_shape": output_shape,
            "output_dtype": str(np.dtype(output_details["dtype"])),
            "label_count": len(labels),
        },
        "artifacts": {
            "model_tflite": to_rel_path(args.model_tflite),
            "labels": to_rel_path(args.labels),
            "dataset": to_rel_path(args.dataset),
        },
        "labels": labels,
        "cases": cases,
        "no_hand_cases": no_hand_cases,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(parity_pack, indent=2, ensure_ascii=False), encoding="utf-8")

    # Smaller file for Flutter tests: only deterministic model I/O vectors.
    flutter_cases_path = output_path.with_name(output_path.stem + "_flutter_cases.json")
    flutter_cases = {
        "contract": parity_pack["contract"],
        "artifacts": parity_pack["artifacts"],
        "labels": labels,
        "cases": [
            {
                "id": case["id"],
                "input": case["features_63"],
                "expected_index": case["expected_index"],
                "expected_label": case["expected_label"],
                "predicted_index": case["predicted_index"],
                "predicted_label": case["predicted_label"],
                "confidence": case["confidence"],
            }
            for case in cases
        ],
    }
    flutter_cases_path.write_text(json.dumps(flutter_cases, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Parity pack generated")
    print(f"- Main JSON: {to_rel_path(str(output_path))}")
    print(f"- Flutter JSON: {to_rel_path(str(flutter_cases_path))}")
    print(
        "- Summary: "
        f"usable={usable_images}/{total_images}, no_hand={no_hand_images}, accuracy={accuracy * 100:.2f}%"
    )
    print(
        "- Contract: "
        f"input={input_shape} {np.dtype(input_details['dtype'])}, "
        f"output={output_shape} {np.dtype(output_details['dtype'])}, labels={len(labels)}"
    )


if __name__ == "__main__":
    main()
