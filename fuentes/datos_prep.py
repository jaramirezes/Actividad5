import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def cargar_y_limpiar_datos(path_raw='datos/datos_ini/car_insurance_claim.csv'):
    print("--- 1. Carga y Preprocesamiento de Datos ---")
    if not os.path.exists(path_raw):
        raise FileNotFoundError(f"No se encontró el archivo en {path_raw}. Coloca tu CSV en datos/datos_ini/")
        
    df = pd.read_csv(path_raw)
    
    # Limpieza de símbolos monetarios
    money_cols = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
    for col in money_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

    # Variables predictoras y objetivo (CLAIM_FLAG)
    feature_cols = [
        'AGE', 'HOMEKIDS', 'YOJ', 'INCOME', 'TRAVTIME', 
        'BLUEBOOK', 'TIF', 'OLDCLAIM', 'CLM_FREQ', 'MVR_PTS', 
        'CAR_AGE', 'PARENT1', 'MSTATUS', 'GENDER', 'EDUCATION', 
        'OCCUPATION', 'CAR_USE', 'CAR_TYPE', 'RED_CAR', 'REVOKED', 'URBANICITY'
    ]
    target_col = 'CLAIM_FLAG'
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols]
    y = df[target_col]
    
    # Guardar copia limpia
    os.makedirs('datos/datos_limp', exist_ok=True)
    df_clean = pd.concat([X, y], axis=1)
    df_clean.to_csv('datos/datos_limp/car_insurance_claim_clean.csv', index=False)
    print("Dataset limpio guardado en 'datos/datos_limp/'")
    
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

def obtener_preprocesador(X_train):
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns

    return ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])