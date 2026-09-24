import os

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
)


# ==========================================================
# DEFINIÇÃO DAS FEATURES POR GRUPO
# ==========================================================

# Grupo 1: Enxuto
num_enxuto = ['Humidity3pm', 'Pressure3pm', 'WindGustSpeed', 'MaxTemp']
log_enxuto = ['Rainfall']
cat_enxuto = ['RainToday']

# Grupo 2: Intermediário
num_inter = ['Humidity3pm', 'Humidity9am', 'Pressure3pm', 'WindGustSpeed', 'WindSpeed3pm', 'MaxTemp', 'MinTemp']
log_inter = ['Rainfall']
cat_inter = ['RainToday', 'WindGustDir']

# Grupo 3: Completo Otimizado
num_comp_otimizada = ['MinTemp', 'MaxTemp', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                      'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
log_comp_otimizada = ['Rainfall']
cat_nom_comp_otimizada = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

# Grupo 4: Completo
num_comp_nulos = num_comp_otimizada + ['Evaporation', 'Sunshine']
log_comp_nulos = ['Rainfall']
cat_nom_comp_nulos = cat_nom_comp_otimizada.copy()
cat_ord_comp_nulos = ['Cloud9am', 'Cloud3pm']


# ==========================================================
# CRIAÇÃO DOS TRANSFORMERS
# ==========================================================

num_robust_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

num_log_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
    ('log_transform', FunctionTransformer(np.log1p, validate=False)),
    ('scaler', RobustScaler())
])

cat_nominal_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

cat_ordinal_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('encoder', OrdinalEncoder(categories=[[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9]],
                               handle_unknown='use_encoded_value', unknown_value=-1))
])


# ==========================================================
# MONTAGEM DAS PIPELINES (ColumnTransformers)
# ==========================================================

preprocessor_enxuto = ColumnTransformer([
    ('num_robust', num_robust_pipe, num_enxuto),
    ('num_log', num_log_pipe, log_enxuto),
    ('cat_nom', cat_nominal_pipe, cat_enxuto)
], remainder='drop')

preprocessor_intermediario = ColumnTransformer([
    ('num_robust', num_robust_pipe, num_inter),
    ('num_log', num_log_pipe, log_inter),
    ('cat_nom', cat_nominal_pipe, cat_inter)
], remainder='drop')

preprocessor_completo_otimizado = ColumnTransformer([
    ('num_robust', num_robust_pipe, num_comp_otimizada),
    ('num_log', num_log_pipe, log_comp_otimizada),
    ('cat_nom', cat_nominal_pipe, cat_nom_comp_otimizada)
], remainder='drop')

preprocessor_completo_nulos = ColumnTransformer([
    ('num_robust', num_robust_pipe, num_comp_nulos),
    ('num_log', num_log_pipe, log_comp_nulos),
    ('cat_nom', cat_nominal_pipe, cat_nom_comp_nulos),
    ('cat_ord', cat_ordinal_pipe, cat_ord_comp_nulos)
], remainder='drop')


# ==========================================================
# TRATAMENTO BASE (compartilhado entre histórico e OOT)
# ==========================================================

def _caminho_padrao():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'weatherAUS.csv'
    )


def _tratamento_base(df):
    """Aplica os tratamentos comuns antes do corte temporal."""
    df = df.dropna(subset=['RainTomorrow']).copy().reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df = df.drop_duplicates()
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df


def carregar_historico(caminho_raw=None):
    """
    Carrega o weatherAUS.csv bruto, aplica todos os tratamentos
    e retorna apenas o conjunto histórico (antes de 2017-01-01).

    RainToday permanece como string ('Yes'/'No') — o encoding é feito
    pelo OneHotEncoder dentro de cada ColumnTransformer.
    """
    df = pd.read_csv(caminho_raw or _caminho_padrao())
    df = _tratamento_base(df)
    df = df[df['Date'] < '2017-01-01'].copy().reset_index(drop=True)
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    return df


def carregar_oot(caminho_raw=None):
    """
    Carrega o weatherAUS.csv bruto, aplica todos os tratamentos
    e retorna apenas o conjunto OOT (2017-01-01 em diante).

    RainToday permanece como string ('Yes'/'No') — o encoding é feito
    pelo OneHotEncoder dentro de cada ColumnTransformer.
    """
    df = pd.read_csv(caminho_raw or _caminho_padrao())
    df = _tratamento_base(df)
    df = df[df['Date'] >= '2017-01-01'].copy().reset_index(drop=True)
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    return df
