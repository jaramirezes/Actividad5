import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def entrenar_y_registrar_modelos(X, y, X_train, y_train):

    """
    Configura el experimento en MLflow, entrena Logistic Regression y Random Forest
    usando validación cruzada (K-Fold) y registra parámetros, métricas y artefactos.
    """
    # Configurar el experimento en MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Pipeline_Clientes_MLflow")

    # ==========================================
    # 1. MODELO 1: LOGISTIC REGRESSION (con K-Fold CV)
    # ==========================================
    model_name_lr = "Logistic_Regression"

    with mlflow.start_run(run_name=model_name_lr):
        C_param = 1.0
        max_iter_param = 1000

        # Registrar parámetros
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C_param)
        mlflow.log_param("max_iter", max_iter_param)
        mlflow.log_param("cv_folds", 3)

        # Definir modelo
        clf = LogisticRegression(C=C_param, max_iter=max_iter_param, random_state=42)

        # Ejecutar Validación Cruzada (K-Fold, cv=3)
        cv_scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        mean_accuracy = cv_scores.mean()

        # Entrenar en el conjunto completo para guardar el artefacto
        clf.fit(X_train, y_train)

        # Registrar métricas de validación cruzada en MLflow
        mlflow.log_metric("cv_accuracy_mean", mean_accuracy)
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())

        # Guardar artefacto
        mlflow.sklearn.log_model(clf, artifact_path="model")
        print(f"Logistic Regression - K-Fold Accuracy Promedio: {mean_accuracy:.4f}")

    # ==========================================
    # 2. MODELO 2: RANDOM FOREST (con K-Fold CV)
    # ==========================================
    model_name_rf = "Random_Forest"

    with mlflow.start_run(run_name=model_name_rf):
        n_estimators_param = 100
        max_depth_param = 10

        # Registrar parámetros
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators_param)
        mlflow.log_param("max_depth", max_depth_param)
        mlflow.log_param("cv_folds", 3)

        # Definir modelo
        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators_param,
            max_depth=max_depth_param,
            random_state=42
        )

        # Ejecutar Validación Cruzada (K-Fold, cv=3)
        cv_scores_rf = cross_val_score(rf_clf, X, y, cv=3, scoring='accuracy')
        mean_accuracy_rf = cv_scores_rf.mean()

        # Entrenar en el conjunto completo para guardar el artefacto
        rf_clf.fit(X_train, y_train)

        # Registrar métricas de validación cruzada en MLflow
        mlflow.log_metric("cv_accuracy_mean", mean_accuracy_rf)
        mlflow.log_metric("cv_accuracy_std", cv_scores_rf.std())

        # Guardar artefacto
        mlflow.sklearn.log_model(
            rf_clf,
            artifact_path="model",
            serialization_format="cloudpickle"
        )
        print(f"Random Forest - K-Fold Accuracy Promedio: {mean_accuracy_rf:.4f}")

    print("\nModelos evaluados con K-Fold Cross-Validation y registrados en MLflow exitosamente")

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # AGREGADO AL FINAL: Cálculo de métricas adicionales de clasificación
    # -------------------------------------------------------------------------
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    print("\n--- REPORTE DE MÉTRICAS ADICIONALES DE CLASIFICACIÓN ---")

    for nombre, modelo_inst in [("Logistic_Regression", clf), ("Random_Forest", rf_clf)]:
        # Predicciones por validación cruzada para métricas imparciales
        y_pred_cv = cross_val_predict(modelo_inst, X, y, cv=3)

        acc = accuracy_score(y, y_pred_cv)
        prec = precision_score(y, y_pred_cv, average='weighted', zero_division=0)
        rec = recall_score(y, y_pred_cv, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred_cv, average='weighted', zero_division=0)

        try:
            y_prob_cv = cross_val_predict(modelo_inst, X, y, cv=3, method="predict_proba")[:, 1]
            roc_auc = roc_auc_score(y, y_prob_cv)
        except Exception:
            roc_auc = 0.0

        print(f"\nModelo: {nombre}")
        print(f"  • Accuracy:  {acc:.4f}")
        print(f"  • Precision: {prec:.4f}")
        print(f"  • Recall:    {rec:.4f}")
        print(f"  • F1-Score:  {f1:.4f}")
        print(f"  • ROC-AUC:   {roc_auc:.4f}")
