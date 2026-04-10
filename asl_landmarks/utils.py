import os
import urllib.request

DEFAULT_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MOVING_CLASSES = {"J", "Z"}


def ensure_hand_landmarker(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        return path
    print(f"Descargando hand_landmarker.task a {path}...")
    urllib.request.urlretrieve(DEFAULT_HAND_LANDMARKER_URL, path)
    return path


def parse_csv_classes(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None
