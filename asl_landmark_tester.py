"""Pruebas locales para modelo ASL de landmarks sin Flutter."""

import argparse
import csv
import os

import cv2
import numpy as np
import tensorflow as tf

from asl_landmarks.extractor import LandmarkExtractor
from asl_landmarks.utils import VALID_EXTENSIONS, ensure_hand_landmarker


def load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def predict_topk(
    model: tf.keras.Model,
    extractor: LandmarkExtractor,
    image_bgr: np.ndarray,
    labels: list[str],
    top_k: int,
):
    sample = extractor.extract_from_bgr(image_bgr)
    if sample is None:
        return None

    probabilities = model.predict(np.expand_dims(sample.features, axis=0), verbose=0)[0]
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    return [(labels[idx], float(probabilities[idx])) for idx in top_indices]


def run_image_mode(args, model, extractor, labels):
    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer imagen: {args.input}")

    predictions = predict_topk(model, extractor, image, labels, args.top_k)
    if predictions is None:
        print("No se detecto mano en la imagen")
        return

    print("Predicciones:")
    for class_name, prob in predictions:
        print(f"- {class_name}: {prob * 100:.2f}%")


def run_folder_mode(args, model, extractor, labels):
    files = [
        name
        for name in sorted(os.listdir(args.input))
        if name.lower().endswith(VALID_EXTENSIONS)
    ]

    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print("No se encontraron imagenes validas")
        return

    results = []
    detected = 0

    for name in files:
        image_path = os.path.join(args.input, name)
        image = cv2.imread(image_path)
        if image is None:
            results.append((name, "read_error", 0.0))
            print(f"[SKIP] {name}: lectura")
            continue

        predictions = predict_topk(model, extractor, image, labels, args.top_k)
        if predictions is None:
            results.append((name, "no_hand", 0.0))
            print(f"[NO_HAND] {name}")
            continue

        detected += 1
        top_class, top_prob = predictions[0]
        results.append((name, top_class, top_prob))
        print(f"[OK] {name}: {top_class} ({top_prob * 100:.2f}%)")

    print(f"\nResumen con mano detectada: {detected}/{len(files)}")

    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(["filename", "prediction", "confidence"])
            writer.writerows(results)
        print(f"CSV guardado en: {args.csv_out}")


def run_webcam_mode(args, model, extractor, labels):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return

    print("Webcam activa. Teclas: q=salir, p=pausa")
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            predictions = predict_topk(model, extractor, frame, labels, args.top_k)
            if predictions:
                for row, (class_name, prob) in enumerate(predictions):
                    y = 30 + (row * 30)
                    color = (0, 255, 0) if row == 0 else (255, 255, 255)
                    cv2.putText(
                        frame,
                        f"{class_name}: {prob * 100:.1f}%",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

            cv2.imshow("ASL Landmark Tester", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tester local ASL por landmarks")
    parser.add_argument("--mode", choices=["image", "folder", "webcam"], default="image")
    parser.add_argument("--model", default="artifacts/landmarks/asl_landmark_model.keras")
    parser.add_argument("--labels", default="artifacts/landmarks/labels.txt")
    parser.add_argument("--input", default=None, help="Imagen o carpeta segun modo")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--csv_out", default=None, help="Solo modo folder")
    parser.add_argument("--hand_model", default=".cache/models/hand_landmarker.task")
    parser.add_argument("--min_detection_confidence", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"No existe modelo: {args.model}")
    if not os.path.isfile(args.labels):
        raise FileNotFoundError(f"No existe labels: {args.labels}")
    if args.mode in {"image", "folder"} and not args.input:
        raise ValueError("Debes pasar --input en modo image/folder")

    ensure_hand_landmarker(args.hand_model)

    model = tf.keras.models.load_model(args.model)
    labels = load_labels(args.labels)
    extractor = LandmarkExtractor(
        model_asset_path=args.hand_model,
        min_detection_confidence=args.min_detection_confidence,
    )

    if args.mode == "image":
        run_image_mode(args, model, extractor, labels)
    elif args.mode == "folder":
        run_folder_mode(args, model, extractor, labels)
    else:
        run_webcam_mode(args, model, extractor, labels)


if __name__ == "__main__":
    main()
