"""CLI de entrenamiento ASL basado en landmarks (pipeline modular)."""

import argparse

from asl_landmarks.config import TrainingConfig
from asl_landmarks.pipeline import train_landmark_model
from asl_landmarks.utils import parse_csv_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento ASL con landmarks")
    parser.add_argument("--dataset", default="archive/asl_alphabet_train/asl_alphabet_train")
    parser.add_argument("--output_dir", default="artifacts/landmarks")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--validation_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images_per_class", type=int, default=None)
    parser.add_argument("--min_detection_confidence", type=float, default=0.3)
    parser.add_argument("--min_samples_per_class", type=int, default=5)
    parser.add_argument("--class_mode", choices=["all", "static", "dynamic"], default="all")
    parser.add_argument("--include_classes", default=None, help="Ej: A,B,C")
    parser.add_argument("--exclude_classes", default=None, help="Ej: nothing,space")
    parser.add_argument("--quantization", choices=["none", "dynamic", "float16"], default="float16")
    parser.add_argument("--augment_noise_std", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TrainingConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        test_split=args.test_split,
        random_seed=args.seed,
        max_images_per_class=args.max_images_per_class,
        include_classes=parse_csv_classes(args.include_classes),
        exclude_classes=parse_csv_classes(args.exclude_classes),
        min_detection_confidence=args.min_detection_confidence,
        min_samples_per_class=args.min_samples_per_class,
        class_mode=args.class_mode,
        quantization=args.quantization,
        augment_noise_std=args.augment_noise_std,
    )

    artifacts = train_landmark_model(config)
    print("\nArchivos generados:")
    print(f"- Modelo keras: {artifacts.keras_model_path}")
    print(f"- Modelo tflite: {artifacts.tflite_model_path}")
    print(f"- Labels: {artifacts.labels_path}")
    print(f"- Metadata: {artifacts.metadata_path}")
    print(f"- Reporte: {artifacts.report_path}")
    print(f"- Matriz confusion: {artifacts.confusion_matrix_path}")


if __name__ == "__main__":
    main()
