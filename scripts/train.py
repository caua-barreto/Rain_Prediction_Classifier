"""
Reproduz o treinamento final do modelo campeão (Completa c/ Nulos),
avalia nas bases histórica e OOT, e salva os artefatos em artifacts/.
"""
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (
    carregar_historico,
    carregar_oot,
    preprocessor_completo_nulos,
)

TARGET = 'RainTomorrow'
RANDOM_STATE = 14
MAX_ITER = 1000
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts'
)


def _ks_statistic(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    all_sorted = np.sort(y_prob)
    cdf_pos = np.searchsorted(pos_sorted, all_sorted, side='right') / len(pos)
    cdf_neg = np.searchsorted(neg_sorted, all_sorted, side='right') / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def _calcular_metricas(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'recall': round(recall_score(y_true, y_pred), 4),
        'f1_score': round(f1_score(y_true, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_true, y_prob), 4),
        'pr_auc': round(average_precision_score(y_true, y_prob), 4),
        'ks': round(_ks_statistic(y_true, y_prob), 4),
    }


def _otimizar_threshold(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (
        2 * precisions[:-1] * recalls[:-1]
        / (precisions[:-1] + recalls[:-1] + 1e-10)
    )
    melhor_idx = np.argmax(f1_scores)
    return float(thresholds[melhor_idx])


def main():
    print("=" * 60)
    print(" CARREGANDO DADOS")
    print("=" * 60)

    df_historico = carregar_historico()
    df_oot = carregar_oot()

    X_hist = df_historico.drop(columns=[TARGET])
    y_hist = df_historico[TARGET]

    X_oot = df_oot.drop(columns=[TARGET])
    y_oot = df_oot[TARGET]

    print(f"  Histórico: {X_hist.shape[0]:,} linhas")
    print(f"  OOT (2017): {X_oot.shape[0]:,} linhas")

    print("\n" + "=" * 60)
    print(" TREINANDO MODELO: Completa c/ Nulos")
    print("=" * 60)

    modelo = Pipeline([
        ('prep', preprocessor_completo_nulos),
        ('clf', LogisticRegression(
            solver='liblinear',
            penalty='l1',
            C=93.95337064926001,
            class_weight=None,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
        ))
    ])

    modelo.fit(X_hist, y_hist)

    print("  Modelo treinado com 100% dos dados históricos.")

    print("\n" + "=" * 60)
    print(" OTIMIZANDO THRESHOLD")
    print("=" * 60)

    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_probs_cv = cross_val_predict(
        modelo, X_hist, y_hist, cv=skf, method='predict_proba', n_jobs=-1
    )[:, 1]

    threshold = _otimizar_threshold(y_hist, y_probs_cv)
    print(f"  Threshold ótimo (max F1): {threshold:.4f}")

    print("\n" + "=" * 60)
    print(" AVALIANDO MODELO")
    print("=" * 60)

    y_prob_hist = modelo.predict_proba(X_hist)[:, 1]
    metrics_test = _calcular_metricas(y_hist, y_prob_hist, threshold)

    y_prob_oot = modelo.predict_proba(X_oot)[:, 1]
    metrics_oot = _calcular_metricas(y_oot, y_prob_oot, threshold)

    print("\n  Métricas Histórico (teste):")
    for k, v in metrics_test.items():
        print(f"    {k:>12s}: {v:.4f}")

    print("\n  Métricas OOT:")
    for k, v in metrics_oot.items():
        print(f"    {k:>12s}: {v:.4f}")

    print("\n" + "=" * 60)
    print(" SALVANDO ARTEFATOS")
    print("=" * 60)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    artefato = {
        'modelo_nome': 'Completa c/ Nulos',
        'modelo': modelo,
        'threshold': threshold,
    }
    caminho_modelo = os.path.join(ARTIFACTS_DIR, 'modelo.joblib')
    joblib.dump(artefato, caminho_modelo)
    print(f"  modelo.joblib  -> {caminho_modelo}")

    metrics_json = {
        'threshold': round(threshold, 4),
        'test': metrics_test,
        'oot': metrics_oot,
    }
    caminho_metrics = os.path.join(ARTIFACTS_DIR, 'metrics.json')
    with open(caminho_metrics, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  metrics.json   -> {caminho_metrics}")

    metadata = {
        'target': TARGET,
        'model_name': 'Completa c/ Nulos',
        'model_class': 'LogisticRegression',
        'hyperparameters': {
            'solver': 'liblinear',
            'penalty': 'l1',
            'C': 93.95337064926001,
            'class_weight': None,
            'max_iter': MAX_ITER,
            'random_state': RANDOM_STATE,
        },
        'threshold_optimization': {
            'method': 'max_f1_via_precision_recall_curve',
            'cv_folds': 5,
        },
        'features': {
            'numeric_robust': [
                'MinTemp', 'MaxTemp', 'WindGustSpeed', 'WindSpeed9am',
                'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm',
                'Sunshine', 'Evaporation',
            ],
            'numeric_log1p': ['Rainfall'],
            'categorical_nominal': [
                'Location', 'WindGustDir', 'WindDir9am',
                'WindDir3pm', 'RainToday',
            ],
            'categorical_ordinal': ['Cloud9am', 'Cloud3pm'],
        },
        'temporal_cutoff': '2017-01-01',
        'historical_size': int(X_hist.shape[0]),
        'oot_size': int(X_oot.shape[0]),
        'target_rate_historical': round(float(y_hist.mean()), 4),
        'target_rate_oot': round(float(y_oot.mean()), 4),
        'dataset': 'weatherAUS',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    caminho_metadata = os.path.join(ARTIFACTS_DIR, 'metadata.json')
    with open(caminho_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json  -> {caminho_metadata}")

    print("\n" + "=" * 60)
    print(" TREINAMENTO CONCLUÍDO")
    print("=" * 60)


if __name__ == '__main__':
    main()
