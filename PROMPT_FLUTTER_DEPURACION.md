# Prompt de depuracion Flutter (ASL landmarks)

## Prompt

Actua como ingeniero senior Flutter + ML y ayudame a depurar por que la app no reconoce bien letras ASL aun usando un modelo TFLite funcional.

Contexto fijo:
- Repo de entrenamiento: Entrenamiento-ASL
- Modelo a usar: artifacts/landmarks_full/asl_landmark_model.tflite
- Labels: artifacts/landmarks_full/labels.txt
- Metadata: artifacts/landmarks_full/model_metadata.json
- Parity pack generado en Python:
  - artifacts/flutter_parity/landmarks_full_parity.json
  - artifacts/flutter_parity/landmarks_full_parity_flutter_cases.json

Hipotesis principal:
El fallo esta en pipeline de Flutter (preprocesamiento, shape/dtype, labels o lectura de output), no en entrenamiento.

Tareas obligatorias:
1. Revisar y corregir contrato TFLite en Flutter:
   - input [1,63], float32
   - output [1,num_classes], float32
   - labels length == num_classes
2. Implementar logs de depuracion para cada inferencia:
   - handedness detectada
   - primeros 8 valores del vector de 63
   - norma maxima antes de normalizar
   - top-3 clases con probabilidad
3. Verificar preprocesamiento exacto:
   - 21 landmarks x,y,z
   - si left: x = -x
   - restar wrist (landmark 0)
   - dividir por norma maxima (si > 1e-6)
   - flatten a 63 float32
4. Cargar landmarks_full_parity_flutter_cases.json y ejecutar un test local:
   - para cada input del JSON, correr interpreter
   - comparar top-1 contra predicted_index del JSON
   - reportar porcentaje de coincidencia
5. Si parity por vector pasa pero camara falla:
   - localizar divergencia en etapa de landmarks
   - ajustar thresholds y suavizado temporal (ventana 5)
6. Entregar cambios listos para correr con pasos exactos.

Entregables:
- Archivos Dart modificados
- Test o script de parity en Flutter
- Resumen de causa raiz
- Checklist de validacion final en dispositivo

Reglas:
- No mezclar artefactos de distintas carpetas.
- No cambiar orden de labels.
- No asumir modelo de imagen 224x224 (este modelo es landmarks).
- Si hay incertidumbre, mostrar datos instrumentados antes de proponer cambios.
