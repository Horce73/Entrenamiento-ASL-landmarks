import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def save_training_plots(history: Any, output_dir: str) -> str:
    output_path = os.path.join(output_dir, "training_history.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history.get("accuracy", []), label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history.get("loss", []), label="train")
    axes[1].plot(history.history.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], output_dir: str) -> str:
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def save_labels(class_names: list[str], output_dir: str) -> str:
    output_path = os.path.join(output_dir, "labels.txt")
    with open(output_path, "w", encoding="utf-8") as file_obj:
        for name in class_names:
            file_obj.write(f"{name}\n")
    return output_path


def save_metadata(metadata: dict[str, Any], output_dir: str) -> str:
    output_path = os.path.join(output_dir, "model_metadata.json")
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=False)
    return output_path


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], output_dir: str) -> str:
    output_path = os.path.join(output_dir, "classification_report.json")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False)
    return output_path
