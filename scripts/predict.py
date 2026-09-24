"""
Carrega o pipeline salvo em artifacts/modelo.joblib, recebe o caminho
de um novo arquivo .csv e gera as probabilidades da classe positiva
(RainTomorrow = Yes).

Uso:
    python scripts/predict.py --input data/new.csv
    python scripts/predict.py --input data/new.csv --output predictions.csv
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts'
)

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports'
)

def carregar_artefatos():
    caminho_modelo = os.path.join(ARTIFACTS_DIR, 'modelo.joblib')
    caminho_metadata = os.path.join(ARTIFACTS_DIR, 'metadata.json')

    artefato = joblib.load(caminho_modelo)
    modelo = artefato['modelo']

    metadata = None
    if os.path.exists(caminho_metadata):
        with open(caminho_metadata) as f:
            metadata = json.load(f)

    return modelo, metadata


def preprocess(df_raw):
    df = df_raw.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
    return df


def prever(caminho_csv):
    modelo, metadata = carregar_artefatos()

    print("=" * 60)
    print(" PREDIÇÃO — RainTomorrow")
    print("=" * 60)
    print(f"  Modelo  : {metadata['model_name'] if metadata else 'N/A'}")
    print(f"  Arquivo : {caminho_csv}")

    df_raw = pd.read_csv(caminho_csv)
    df = preprocess(df_raw)

    if 'RainTomorrow' in df.columns:
        df = df.drop(columns=['RainTomorrow'])

    y_prob = modelo.predict_proba(df)[:, 1]

    resultado = pd.DataFrame({
        'rain_probability': np.round(y_prob, 4),
    })

    if 'Date' in df_raw.columns:
        resultado.insert(0, 'Date', df_raw['Date'])
    if 'Location' in df_raw.columns:
        resultado.insert(1 if 'Date' in df_raw.columns else 0, 'Location', df_raw['Location'])

    print(f"\n  Total de registros   : {len(resultado)}")
    print("=" * 60)

    return resultado


def main():
    parser = argparse.ArgumentParser(
        description='Gera probabilidades de RainTomorrow para um CSV novo.'
    )
    parser.add_argument('--input', required=True, help='Caminho do CSV de entrada')
    parser.add_argument('--output', default=None,
                        help='Caminho do CSV de saída (padrão: reports/<nome>_predictions.csv)')

    args = parser.parse_args()

    resultado = prever(args.input)

    if args.output:
        caminho_saida = args.output
    else:
        nome_entrada = os.path.splitext(os.path.basename(args.input))[0]
        caminho_saida = os.path.join(REPORTS_DIR, f'{nome_entrada}_predictions.csv')

    os.makedirs(os.path.dirname(os.path.abspath(caminho_saida)), exist_ok=True)
    resultado.to_csv(caminho_saida, index=False)
    print(f"\n  Resultado salvo em: {caminho_saida}")


if __name__ == '__main__':
    main()
