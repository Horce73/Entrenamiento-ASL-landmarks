import json
import os
import unittest

import numpy as np
import tensorflow as tf


class TestLandmarkModelContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_dir = os.environ.get("ASL_MODEL_DIR", "artifacts/landmarks_smoke")
        cls.model_dir = model_dir
        cls.model_path = os.path.join(model_dir, "asl_landmark_model.keras")
        cls.labels_path = os.path.join(model_dir, "labels.txt")
        cls.metadata_path = os.path.join(model_dir, "model_metadata.json")
        cls.summary_path = os.path.join(model_dir, "training_summary.json")
        cls.report_path = os.path.join(model_dir, "classification_report.json")

    def test_01_required_artifacts_exist(self):
        required = [
            self.model_path,
            self.labels_path,
            self.metadata_path,
            self.summary_path,
            self.report_path,
        ]
        missing = [path for path in required if not os.path.isfile(path)]
        self.assertEqual(missing, [], f"Faltan artefactos: {missing}")

    def test_02_metadata_matches_labels(self):
        with open(self.labels_path, "r", encoding="utf-8") as file_obj:
            labels = [line.strip() for line in file_obj if line.strip()]

        with open(self.metadata_path, "r", encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)

        self.assertEqual(metadata["num_classes"], len(labels))
        self.assertEqual(metadata["class_names"], labels)
        self.assertEqual(metadata["feature_size"], 63)

    def test_03_model_shape_contract(self):
        model = tf.keras.models.load_model(self.model_path)

        input_shape = model.input_shape
        output_shape = model.output_shape

        self.assertEqual(input_shape[-1], 63, "Input esperado: vector landmarks de 63")

        with open(self.labels_path, "r", encoding="utf-8") as file_obj:
            labels = [line.strip() for line in file_obj if line.strip()]
        self.assertEqual(output_shape[-1], len(labels), "Output debe coincidir con labels")

    def test_04_inference_probability_contract(self):
        model = tf.keras.models.load_model(self.model_path)
        fake_sample = np.zeros((1, 63), dtype=np.float32)

        probs = model.predict(fake_sample, verbose=0)[0]

        self.assertTrue(np.isfinite(probs).all(), "Predicciones contienen NaN o Inf")
        self.assertTrue(np.all(probs >= 0.0), "Probabilidades negativas")
        self.assertTrue(np.all(probs <= 1.0), "Probabilidades mayores que 1")
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=3)

    def test_05_minimum_accuracy_gate(self):
        min_accuracy = float(os.environ.get("ASL_MIN_TEST_ACCURACY", "0.35"))
        with open(self.summary_path, "r", encoding="utf-8") as file_obj:
            summary = json.load(file_obj)

        actual_accuracy = float(summary["metrics"]["test_accuracy"])
        self.assertGreaterEqual(
            actual_accuracy,
            min_accuracy,
            f"Accuracy {actual_accuracy:.4f} por debajo del minimo {min_accuracy:.4f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
