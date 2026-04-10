"""CLI para probar reconocimiento ASL con deteccion de mano MediaPipe en vivo."""

import argparse

from asl_landmarks.live_probe import LiveMediapipeProbe, LiveProbeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe ASL con MediaPipe en webcam")
    parser.add_argument("--model", default="artifacts/landmarks_smoke/asl_landmark_model.keras")
    parser.add_argument("--labels", default="artifacts/landmarks_smoke/labels.txt")
    parser.add_argument("--hand_model", default=".cache/models/hand_landmarker.task")
    parser.add_argument("--min_detection_confidence", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--camera_index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LiveProbeConfig(
        model_path=args.model,
        labels_path=args.labels,
        hand_model_path=args.hand_model,
        min_detection_confidence=args.min_detection_confidence,
        top_k=args.top_k,
        camera_index=args.camera_index,
    )

    probe = LiveMediapipeProbe(config)
    probe.run()


if __name__ == "__main__":
    main()
