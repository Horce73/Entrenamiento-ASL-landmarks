# Prompt de integracion Flutter (ASL landmarks)

Copia y pega este prompt en Copilot Chat cuando estes trabajando entre este repo y tu app Flutter.

## Prompt

Actua como ingeniero senior de ML + Flutter y ayudame a integrar el modelo ASL de landmarks exportado desde este repositorio.

Contexto del repo actual:
- Modelo recomendado: artifacts/landmarks_full/asl_landmark_model.tflite
- Labels recomendados: artifacts/landmarks_full/labels.txt
- Metadata recomendada: artifacts/landmarks_full/model_metadata.json
- Contrato esperado del modelo:
  - input shape: [1, 63]
  - input dtype: float32
  - output shape: [1, num_classes]
  - output dtype: float32
- Normalizacion esperada: wrist_centered_maxnorm_handedness_canonical

Objetivo:
Integrar inferencia robusta en Flutter replicando exactamente el preprocesamiento de Python para evitar drift entre entrenamiento y app movil.

Requisitos obligatorios:
1. No entrenar nada en TensorFlow Lite. Solo inferencia en TFLite.
2. Usar un unico bundle coherente (modelo + labels + metadata de la misma carpeta).
3. Implementar preprocesamiento de landmarks en Flutter exactamente asi:
   - detectar 21 landmarks con x,y,z
   - si la mano es left, invertir x (x = -x) para canonizar handedness
   - restar el landmark wrist (indice 0) a todos los puntos
   - calcular norma de cada punto y dividir todo por la norma maxima (si max > 1e-6)
   - aplanar a vector float32 de 63
4. Ejecutar inferencia TFLite con tensor [1, 63] float32.
5. Mapear salida con labels.txt en el mismo orden exacto.
6. Agregar defensas y logs:
   - error claro si cantidad de labels != salida del modelo
   - error claro si input no es 63
   - warning si no hay mano detectada
7. Evitar falsos positivos:
   - umbral de confianza configurable (ej 0.60)
   - suavizado temporal por ventana corta (ej 5 frames) con voto o promedio
8. Entregar cambios listos para ejecutar en Flutter con instrucciones de prueba.

Entregables esperados:
- Codigo Dart de un servicio de inferencia TFLite.
- Codigo Dart de preprocesamiento de landmarks (funcion pura testeable).
- Actualizacion de pubspec.yaml para assets.
- Checklist de validacion manual en dispositivo.
- Paso de smoke test para comparar top-1 entre Python y Flutter con 1-2 muestras.

Restricciones:
- No cambiar el orden de labels.
- No mezclar artefactos de carpetas distintas.
- No asumir imagen RGB 224x224, porque este modelo es de landmarks (63 features), no de imagen directa.

Si encuentras discrepancias, primero diagnostica y reporta:
- shape/dtype de input-output del modelo
- numero de clases en metadata
- numero de labels en labels.txt
- carpeta exacta de donde salio cada artefacto

Al final, dame:
1. Resumen de cambios.
2. Archivos tocados.
3. Comandos para correr y validar.
4. Riesgos pendientes.
