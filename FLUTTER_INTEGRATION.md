# Integración con Flutter (ASL)

Este proyecto ya puede generar los archivos necesarios para una app Flutter:

- `asl_model.tflite`
- `labels.txt`
- `model_metadata.json`

## 1) Entrenar y exportar bundle Flutter en un paso

```bash
source asl_env/bin/activate
python asl_trainer.py \
  --dataset archive/asl_alphabet_train/asl_alphabet_train \
  --model_type mobilenet \
  --epochs 20 \
  --flutter_out flutter_export \
  --quantization float16
```

Salida esperada en `flutter_export/`:

- `asl_model.tflite`
- `labels.txt`
- `model_metadata.json`

## 2) Exportar desde un modelo ya entrenado (sin reentrenar)

Si ya tienes `asl_model.keras`, usa:

```bash
source asl_env/bin/activate
python export_flutter_assets.py \
  --model asl_model.keras \
  --dataset archive/asl_alphabet_train/asl_alphabet_train \
  --out flutter_export \
  --quantization float16
```

## 3) Copiar directo a assets de tu app Flutter

```bash
source asl_env/bin/activate
python export_flutter_assets.py \
  --model asl_model.keras \
  --dataset archive/asl_alphabet_train/asl_alphabet_train \
  --out flutter_export \
  --flutter_assets /home/hache173/Programación/Proyectos/test02/assets
```

## 4) Registrar assets en pubspec.yaml (app Flutter)

```yaml
flutter:
  assets:
    - assets/asl_model.tflite
    - assets/labels.txt
    - assets/model_metadata.json
```

## 5) Nota importante de preprocesamiento

La app Flutter debe replicar exactamente este preprocesamiento:

- Resize a `224x224`
- RGB
- `float32`
- Normalizar con `pixel / 255.0`

Si cambias `--img_size` al entrenar/exportar, debes usar ese mismo tamaño en Flutter.

## 6) Entrenamiento con landmarks (recomendado cuando falla con imagen estática)

Este modo entrena un clasificador sobre landmarks de mano (21 puntos x 3 coordenadas = 63 features) usando MediaPipe.

### Entrenar landmarks + exportar bundle Flutter

```bash
source asl_env/bin/activate
python asl_landmark_trainer.py \
  --dataset archive/asl_alphabet_train/asl_alphabet_train \
  --output_dir artifacts/landmarks \
  --epochs 20 \
  --batch_size 64 \
  --quantization float16
```

Salida esperada en `artifacts/landmarks/`:

- `asl_landmark_model.keras`
- `asl_landmark_model.tflite`
- `labels.txt`
- `model_metadata.json`

### Validar modelo landmarks antes de integrar en Flutter

```bash
source asl_env/bin/activate
python asl_landmark_validator.py \
  --model artifacts/landmarks/asl_landmark_model.keras \
  --labels artifacts/landmarks/labels.txt \
  --dataset archive/asl_alphabet_test/asl_alphabet_test \
  --output_dir artifacts/landmarks_validation
```

Revisa `artifacts/landmarks_validation/validation_report.json` y usa ese accuracy como puerta de calidad antes de publicar en movil.

### Copiar assets landmarks a Flutter

```bash
cp flutter_export_landmarks/asl_landmark_model.tflite /home/hache173/Programación/Proyectos/test02/assets/
cp flutter_export_landmarks/labels.txt /home/hache173/Programación/Proyectos/test02/assets/
cp flutter_export_landmarks/model_metadata.json /home/hache173/Programación/Proyectos/test02/assets/
```

Si usas el nuevo pipeline:

```bash
cp artifacts/landmarks/asl_landmark_model.tflite /home/hache173/Programación/Proyectos/test02/assets/
cp artifacts/landmarks/labels.txt /home/hache173/Programación/Proyectos/test02/assets/
cp artifacts/landmarks/model_metadata.json /home/hache173/Programación/Proyectos/test02/assets/
```

### Registrar assets landmarks en pubspec.yaml

```yaml
flutter:
  assets:
    - assets/asl_landmark_model.tflite
    - assets/labels.txt
    - assets/model_metadata.json
```

### Preprocesamiento que Flutter debe replicar para landmarks

1. Detectar 21 landmarks de mano (x, y, z) por frame con MediaPipe Hands.
2. Si la mano detectada es `left`, invertir eje X (`x = -x`) para canonizar handedness.
3. Restar coordenadas del wrist (landmark 0) a todos los puntos.
4. Dividir por la norma máxima (max distancia al wrist).
5. Aplanar a vector `float32` de tamaño 63.
6. Ejecutar inferencia TFLite con input shape `[1, 63]`.

## 7) Probar modelo landmarks localmente (sin Flutter)

### Probar una imagen

```bash
source asl_env/bin/activate
python asl_landmark_tester.py \
  --mode image \
  --input archive/asl_alphabet_train/asl_alphabet_train/A/A1.jpg \
  --model artifacts/landmarks/asl_landmark_model.keras \
  --labels artifacts/landmarks/labels.txt
```

### Probar carpeta completa

```bash
source asl_env/bin/activate
python asl_landmark_tester.py \
  --mode folder \
  --input archive/asl_alphabet_train/asl_alphabet_train/A \
  --model artifacts/landmarks/asl_landmark_model.keras \
  --labels artifacts/landmarks/labels.txt \
  --limit 100
```

### Probar webcam

```bash
source asl_env/bin/activate
python asl_landmark_tester.py \
  --mode webcam \
  --model artifacts/landmarks/asl_landmark_model.keras \
  --labels artifacts/landmarks/labels.txt
```
  
## 8) Separar entrenamiento de letras estáticas y de movimiento

Letras con movimiento en este flujo: `J`, `Z`.

### Entrenar solo estáticas (excluye J y Z)

```bash
source asl_env/bin/activate
python asl_landmark_trainer.py \
  --dataset archive/asl_alphabet_train/asl_alphabet_train \
  --output_dir artifacts/landmarks_static \
  --class_mode static \
  --epochs 20 \
  --max_images_per_class 1000
```

### Entrenar solo dinámicas (solo J y Z)

```bash
source asl_env/bin/activate
python asl_landmark_trainer.py \
  --dataset archive/asl_alphabet_train/asl_alphabet_train \
  --output_dir artifacts/landmarks_dynamic \
  --class_mode dynamic \
  --epochs 20 \
  --max_images_per_class 1000
```

### Afinar selección manual de clases (opcional)

```bash
python asl_landmark_trainer.py --include_classes A,B,C,J,Z
python asl_landmark_trainer.py --exclude_classes nothing,space
```

## 9) Diagnostico de integracion Flutter (si no reconoce bien)

Cuando la app no reconoce bien letras, primero valida paridad exacta entre Python y Flutter.

### Generar parity pack (input vector + salida esperada)

```bash
source asl_env/bin/activate
python asl_flutter_parity_pack.py \
  --model_tflite artifacts/landmarks_full/asl_landmark_model.tflite \
  --labels artifacts/landmarks_full/labels.txt \
  --dataset archive/asl_alphabet_test/asl_alphabet_test \
  --output artifacts/flutter_parity/landmarks_full_parity.json \
  --max_images_per_class 1
```

Archivos generados:

- `artifacts/flutter_parity/landmarks_full_parity.json`
- `artifacts/flutter_parity/landmarks_full_parity_flutter_cases.json`

### Como usar el parity pack

1. Cargar `landmarks_full_parity_flutter_cases.json` en Flutter.
2. Pasar cada `input` (63 floats) al interpreter TFLite de la app.
3. Verificar que el top-1 coincide con `predicted_index`/`predicted_label` del JSON.

Si falla esta prueba, el problema esta en inferencia Flutter (shape, dtype, labels o lectura de output).
Si pasa esta prueba pero falla en camara, el problema esta en preprocesamiento de landmarks en Flutter.

## 10) Demo rapida para presentacion: escribir con senas en vivo

Si te quedaste sin tiempo y necesitas mostrar un flujo "sena -> texto", usa este demo local:

```bash
source asl_env/bin/activate
python asl_sign_writer_demo.py \
  --model artifacts/landmarks_full/asl_landmark_model.keras \
  --labels artifacts/landmarks_full/labels.txt \
  --confidence_threshold 0.65
```

Controles del demo:

- `q`: salir
- `c`: limpiar texto

Como funciona internamente:

1. Detecta landmarks con MediaPipe.
2. Predice letra con el clasificador de landmarks.
3. Usa ventana temporal (`window_size`) + votos (`min_votes`) para estabilizar la letra.
4. Solo confirma una nueva letra cuando vuelves a estado neutral (evita repetir por frame).
5. Soporta tokens especiales del dataset:
   - `space`: agrega espacio
   - `del`: borra ultimo caracter

Parametros utiles para ajustar en vivo:

```bash
# Menos ruido (mas estricto)
python asl_sign_writer_demo.py --confidence_threshold 0.75 --window_size 9 --min_votes 6

# Mas sensible (si tarda en escribir)
python asl_sign_writer_demo.py --confidence_threshold 0.55 --window_size 5 --min_votes 3
```
