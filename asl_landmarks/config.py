from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    dataset_path: str
    output_dir: str = "artifacts/landmarks"
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    max_images_per_class: Optional[int] = None
    include_classes: Optional[list[str]] = None
    exclude_classes: Optional[list[str]] = None
    min_detection_confidence: float = 0.3
    min_samples_per_class: int = 5
    class_mode: str = "all"
    quantization: str = "float16"
    augment_noise_std: float = 0.01


@dataclass
class ValidationConfig:
    model_path: str
    labels_path: str
    dataset_path: str
    hand_model_path: str = ".cache/models/hand_landmarker.task"
    min_detection_confidence: float = 0.3
    max_images_per_class: Optional[int] = None
    top_k: int = 3
