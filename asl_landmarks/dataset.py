import os
import random
import shutil
from dataclasses import dataclass

import numpy as np

from .config import TrainingConfig
from .extractor import LandmarkExtractor
from .utils import MOVING_CLASSES, VALID_EXTENSIONS


@dataclass
class DatasetBuildResult:
    X: np.ndarray
    y: np.ndarray
    class_names: list[str]
    extraction_stats: dict[str, dict[str, int]]


def _resolve_classes(available: list[str], config: TrainingConfig) -> list[str]:
    if config.class_mode == "static":
        selected = [name for name in available if name not in MOVING_CLASSES]
    elif config.class_mode == "dynamic":
        selected = [name for name in available if name in MOVING_CLASSES]
    elif config.class_mode == "all":
        selected = list(available)
    else:
        raise ValueError("class_mode debe ser all, static o dynamic")

    if config.include_classes:
        include_set = set(config.include_classes)
        selected = [name for name in selected if name in include_set]
    if config.exclude_classes:
        exclude_set = set(config.exclude_classes)
        selected = [name for name in selected if name not in exclude_set]

    return selected


def maybe_build_subset(dataset_path: str, max_images_per_class: int | None, seed: int) -> str:
    if max_images_per_class is None:
        return dataset_path

    subset_root = os.path.join(".cache", f"landmarks_subset_{max_images_per_class}")
    if os.path.exists(subset_root):
        shutil.rmtree(subset_root)
    os.makedirs(subset_root, exist_ok=True)

    rng = random.Random(seed)
    for class_name in sorted(os.listdir(dataset_path)):
        source_class = os.path.join(dataset_path, class_name)
        if not os.path.isdir(source_class):
            continue

        images = [
            name
            for name in os.listdir(source_class)
            if name.lower().endswith(VALID_EXTENSIONS)
        ]
        if not images:
            continue

        rng.shuffle(images)
        selected = images[:max_images_per_class]
        target_class = os.path.join(subset_root, class_name)
        os.makedirs(target_class, exist_ok=True)

        for filename in selected:
            src = os.path.join(source_class, filename)
            dst = os.path.join(target_class, filename)
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)

    return subset_root


def build_landmark_dataset(config: TrainingConfig, extractor: LandmarkExtractor) -> DatasetBuildResult:
    active_dataset = maybe_build_subset(config.dataset_path, config.max_images_per_class, config.random_seed)

    available_classes = [
        name
        for name in sorted(os.listdir(active_dataset))
        if os.path.isdir(os.path.join(active_dataset, name))
    ]
    class_names = _resolve_classes(available_classes, config)
    if not class_names:
        raise RuntimeError("No hay clases luego de aplicar filtros")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    extraction_stats = {name: {"total": 0, "used": 0, "no_hand": 0} for name in class_names}

    X: list[np.ndarray] = []
    y: list[int] = []

    for class_name in class_names:
        class_dir = os.path.join(active_dataset, class_name)
        image_names = [
            name for name in os.listdir(class_dir) if name.lower().endswith(VALID_EXTENSIONS)
        ]
        for image_name in image_names:
            extraction_stats[class_name]["total"] += 1
            sample = extractor.extract_from_path(os.path.join(class_dir, image_name))
            if sample is None:
                extraction_stats[class_name]["no_hand"] += 1
                continue

            X.append(sample.features)
            y.append(class_to_idx[class_name])
            extraction_stats[class_name]["used"] += 1

    if not X:
        raise RuntimeError("No se extrajeron landmarks validos")

    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.int32)

    counts = np.bincount(y_np, minlength=len(class_names))
    valid_indices = [idx for idx, count in enumerate(counts) if count >= config.min_samples_per_class]
    if len(valid_indices) < 2:
        raise RuntimeError("No hay suficientes clases con muestras validas")

    if len(valid_indices) < len(class_names):
        valid_set = set(valid_indices)
        mask = np.asarray([target in valid_set for target in y_np], dtype=bool)
        X_np = X_np[mask]
        y_filtered = y_np[mask]
        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
        y_np = np.asarray([remap[idx] for idx in y_filtered], dtype=np.int32)
        class_names = [class_names[idx] for idx in valid_indices]

    return DatasetBuildResult(X=X_np, y=y_np, class_names=class_names, extraction_stats=extraction_stats)
