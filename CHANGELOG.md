# Changelog - Actividad 5

## [1.1.0] - 2026-10-18

### Actualizado
- **`datos_prep.py`**: Implementación completa del pipeline de preprocesamiento, limpieza de datos, manejo de nulos y codificación de variables.
- **`train.py`**: Integración de la lógica de entrenamiento, validación con Scikit-Learn y seguimiento de experimentos con MLflow.
- **`entrena.ipynb`**: Notebook orquestador para la ejecución interactiva del pipeline completo y visualización de métricas.

## [1.0.0] - 2026-10-16
### Agregado
- Estructura de directorios bajo estándares MLOps (`datos/` y `fuentes/`).
- Integración del dataset original de seguros de auto en `datos/datos_ini/`.
- Script `datos_prep.py` para manejo de valores monetarios, nulos y codificación.
- Script `train.py` con Pipeline de Scikit-Learn, optimización por GridSearch (Logistic Regression vs Random Forest) y métricas trackeadas en MLflow.
- Notebook `entrena.ipynb` para ejecución interactiva.
