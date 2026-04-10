"""Valida un modelo landmarks contra un dataset etiquetado antes de integrarlo en Flutter."""

import argparse

from asl_landmarks.config import ValidationConfig
from asl_landmarks.validation import validate_model_on_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validador pre-movil para ASL landmarks")
    parser.add_argument("--model", default="artifacts/landmarks/asl_landmark_model.keras")
    parser.add_argument("--labels", default="artifacts/landmarks/labels.txt")
    parser.add_argument("--dataset", default="archive/asl_alphabet_test/asl_alphabet_test")
    parser.add_argument("--output_dir", default="artifacts/landmarks_validation")
    parser.add_argument("--hand_model", default=".cache/models/hand_landmarker.task")
    parser.add_argument("--min_detection_confidence", type=float, default=0.3)
    parser.add_argument("--max_images_per_class", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ValidationConfig(
        model_path=args.model,
        labels_path=args.labels,
        dataset_path=args.dataset,
        hand_model_path=args.hand_model,
        min_detection_confidence=args.min_detection_confidence,
        max_images_per_class=args.max_images_per_class,
        top_k=args.top_k,
    )

    summary = validate_model_on_dataset(config, output_dir=args.output_dir)
    print("\nResumen validacion pre-movil:")
    print(f"- Accuracy: {summary.accuracy * 100:.2f}%")
    print(f"- Imagenes totales: {summary.total_images}")
    print(f"- Muestras utilizables: {summary.usable_images}")
    print(f"- Imagenes sin mano: {summary.no_hand_images}")
    print(f"- Reporte: {summary.report_path}")
    print(f"- Matriz confusion: {summary.confusion_path}")


if __name__ == "__main__":
    main()
