# ASL Landmark Training Pipeline

Pipeline modular para entrenar, validar y exportar un modelo de reconocimiento de señas ASL basado en landmarks de mano (21 puntos x 3 coordenadas = 63 features).

## Dataset (Fuente y Creditos)

- Dataset usado: ASL Alphabet (Kaggle)
- Enlace: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- Autor en Kaggle: grassknoted

Este repositorio no incluye el dataset original. Para reproducir entrenamiento y validacion, descargalo desde Kaggle y ubicalo en la estructura esperada.

Nota: revisa y respeta la licencia/términos de uso del dataset en su página de Kaggle antes de redistribuir datos o modelos.

## Que Hace Este Repositorio

- Extrae landmarks con MediaPipe Hands.
- Entrena un clasificador sobre vectores de 63 features.
- Exporta modelos `.keras` y `.tflite`.
- Genera metadata, labels y reportes de calidad.
- Valida pre-integracion movil.
- Incluye utilidades para pruebas locales y parity pack con Flutter.

## Estructura del Proyecto

```text
.
├── asl_landmark_trainer.py
├── asl_landmark_validator.py
├── asl_landmark_tester.py
├── asl_mediapipe_probe.py
├── asl_sign_writer_demo.py
├── asl_flutter_parity_pack.py
├── asl_landmarks/
├── tests/
├── artifacts/
└── archive/
```

## Requisitos

- Linux, macOS o Windows
- Python 3.10+
- Webcam (opcional para modos en vivo)

Dependencias Python en `requirements.txt`.

## Instalacion Rapida

```bash
# 1) Crear entorno virtual
python -m venv asl_env

# 2) Activar entorno
# Linux/macOS:
source asl_env/bin/activate
# Windows (PowerShell):
# .\\asl_env\\Scripts\\Activate.ps1

# 3) Instalar dependencias
pip install -r requirements.txt
```

## Preparar Dataset

Por defecto se usan estas rutas:

- Train: `archive/asl_alphabet_train/asl_alphabet_train`
- Test: `archive/asl_alphabet_test/asl_alphabet_test`

Formato recomendado:

```text
archive/asl_alphabet_train/asl_alphabet_train/
├── A/
├── B/
├── ...
├── Z/
├── del/
├── nothing/
└── space/
```

El validador tambien soporta un formato plano por prefijo, por ejemplo `A_01.jpg`, `space_test.jpg`, etc.

## Flujo Recomendado

### 1) Entrenar

```bash
source asl_env/bin/activate

python asl_landmark_trainer.py \
  --dataset archive/asl_alphabet_train/asl_alphabet_train \
  --output_dir artifacts/landmarks_full \
  --epochs 40 \
  --class_mode all \
  --quantization float16
```

Opciones utiles:

- `--class_mode all|static|dynamic` (dynamic = clases con movimiento como J y Z)
- `--include_classes A,B,C`
- `--exclude_classes nothing,space`
- `--max_images_per_class 1000`

### 2) Validar antes de Flutter

```bash
python asl_landmark_validator.py \
  --model artifacts/landmarks_full/asl_landmark_model.keras \
  --labels artifacts/landmarks_full/labels.txt \
  --dataset archive/asl_alphabet_test/asl_alphabet_test \
  --output_dir artifacts/landmarks_full_validation
```

### 3) Probar localmente

```bash
# Webcam
python asl_landmark_tester.py --mode webcam \
  --model artifacts/landmarks_full/asl_landmark_model.keras \
  --labels artifacts/landmarks_full/labels.txt

# Imagen unica
python asl_landmark_tester.py --mode image \
  --input archive/asl_alphabet_train/asl_alphabet_train/A/A1.jpg \
  --model artifacts/landmarks_full/asl_landmark_model.keras \
  --labels artifacts/landmarks_full/labels.txt
```

## Artefactos Generados

Salida tipica en `artifacts/landmarks_full/`:

- `asl_landmark_model.keras`
- `asl_landmark_model.tflite`
- `labels.txt`
- `model_metadata.json`
- `training_summary.json`
- `classification_report.json`
- `confusion_matrix.png`
- `training_history.png`

Salida tipica de validacion en `artifacts/landmarks_full_validation/`:

- `validation_report.json`
- `validation_confusion_matrix.json`

## Integracion con Flutter

Archivos minimos para Flutter:

- `asl_landmark_model.tflite`
- `labels.txt`
- `model_metadata.json`

Generar parity pack para comparar Python vs Flutter:

```bash
python asl_flutter_parity_pack.py \
  --model_tflite artifacts/landmarks_full/asl_landmark_model.tflite \
  --labels artifacts/landmarks_full/labels.txt \
  --dataset archive/asl_alphabet_test/asl_alphabet_test \
  --output artifacts/flutter_parity/landmarks_full_parity.json \
  --max_images_per_class 1
```

Esto produce:

- `artifacts/flutter_parity/landmarks_full_parity.json`
- `artifacts/flutter_parity/landmarks_full_parity_flutter_cases.json`

## Demos en Vivo

Probe MediaPipe + clasificacion en webcam:

```bash
python asl_mediapipe_probe.py \
  --model artifacts/landmarks_full/asl_landmark_model.keras \
  --labels artifacts/landmarks_full/labels.txt \
  --top_k 3
```

Demo de escritura por señas:

```bash
python asl_sign_writer_demo.py \
  --model artifacts/landmarks_full/asl_landmark_model.keras \
  --labels artifacts/landmarks_full/labels.txt \
  --confidence_threshold 0.65
```

## Pruebas de Contrato del Modelo

El repositorio incluye pruebas para validar consistencia de artefactos y contrato de inferencia:

```bash
source asl_env/bin/activate
python -m unittest discover -s tests -v
```

Opcional:

```bash
# Cambiar directorio de artefactos a validar
ASL_MODEL_DIR=artifacts/landmarks_full python -m unittest discover -s tests -v

# Ajustar gate minimo de accuracy en test
ASL_MIN_TEST_ACCURACY=0.50 python -m unittest discover -s tests -v
```

## Publicar en GitHub

Si aun no publicaste el repositorio:

```bash
git init
git add .
git commit -m "Initial commit: ASL landmarks pipeline"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

## Agradecimientos

- Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- MediaPipe Hands
- TensorFlow / Keras
