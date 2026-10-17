# Changelog - Actividad 5

## [1.0.0] - 2026-10-16
### Agregado
- Estructura de directorios bajo estándares MLOps (`datos/` y `fuentes/`).
- Integración del dataset original de seguros de auto en `datos/datos_ini/`.
- Script `datos_prep.py` para manejo de valores monetarios, nulos y codificación.
- Script `train.py` con Pipeline de Scikit-Learn, optimización por GridSearch (Logistic Regression vs Random Forest) y métricas trackeadas en MLflow.
- Notebook `entrena.ipynb` para ejecución interactiva.