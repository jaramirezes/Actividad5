# Actividad 5: Pipeline MLOps y Comparación de Modelos

Repositorio para la implementación de un pipeline MLOps utilizando el dataset **Car Insurance Claim Data** de Kaggle, comparando **Logistic Regression** y **Random Forest** mediante **Grid Search** y seguimiento de experimentos con **MLflow**.

## Estructura del Repositorio
- `datos/datos_ini/`: Contiene el dataset original (`car_insurance_claim.csv`).
- `datos/datos_limp/`: Contiene el dataset procesado y limpio.
- `fuentes/`: 
  - `entrena.ipynb`: Notebook de desarrollo y evaluación.
  - `datos_prep.py`: Funciones de limpieza e imputación.
  - `train.py`: Script principal de entrenamiento, optimización con GridSearchCV y registro en MLflow.