import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from google.colab import files

def cargar_y_preprocesar_datos(target_col="customer_status", max_unique_categories=50, test_size=0.2, random_state=42):
    print("Por favor, selecciona el dataset a procesar:")
    uploaded = files.upload()

    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(file_name)
    print(f"\nDataset '{file_name}' cargado exitosamente.")

    # 1. Definir target y separar X / y
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # 2. Filtro de limpieza inteligente (alta cardinalidad)
    cols_a_eliminar = [col for col in X.select_dtypes(include=['object']).columns
                       if X[col].nunique() > max_unique_categories]

    if cols_a_eliminar:
        print(f"\nEliminando {len(cols_a_eliminar)} columnas con alta cardinalidad: {cols_a_eliminar}")
        X = X.drop(columns=cols_a_eliminar)

    # 3. Transformación de categóricas e imputación de nulos
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.mean(numeric_only=True))

    # 4. Escalado de datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Guardar y descargar en Colab
    df_limpio_exportar = X.copy()
    df_limpio_exportar[target_col] = y.values

    nombre_archivo_salida = "car_insurance_claim_datos_limpios.csv"
    df_limpio_exportar.to_csv(nombre_archivo_salida, index=False)
    print(f"\n[INFO] Archivo guardado. Descargando automáticamente...")
    files.download(nombre_archivo_salida)

    # 5. División en Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("\n¡Datos preprocesados y divididos exitosamente!")
    return X, y, X_train, X_test, y_train, y_test, df
