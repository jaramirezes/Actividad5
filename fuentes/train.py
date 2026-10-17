import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from datos_prep import cargar_y_limpiar_datos, obtener_preprocesador

mlflow.set_experiment("Actividad5_Comparacion_Modelos")

def ejecutar():
    X_train, X_test, y_train, y_test = cargar_y_limpiar_datos()
    preprocessor = obtener_preprocesador(X_train)

    algoritmos = {
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {'classifier__C': [0.01, 0.1, 1.0]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [5, 10]}
        }
    }

    for nombre, config in algoritmos.items():
        with mlflow.start_run(run_name=f"Run_{nombre}"):
            print(f"\nEntrenando y optimizando: {nombre}...")
            
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', config["model"])])
            grid = GridSearchCV(pipeline, config["params"], cv=5, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            mejor = grid.best_estimator_
            for p, v in grid.best_params_.items(): mlflow.log_param(p, v)
            
            y_pred = mejor.predict(X_test)
            y_prob = mejor.predict_proba(X_test)[:, 1] if hasattr(mejor, "predict_proba") else y_pred

            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
            mlflow.log_metric("recall", recall_score(y_test, y_pred, zero_division=0))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred, zero_division=0))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
            
            mlflow.sklearn.log_model(mejor, f"modelo_{nombre.lower()}")
            print(f"[✔] {nombre} registrado exitosamente en MLflow.")

if __name__ == "__main__":
    ejecutar()