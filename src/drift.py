"""
Módulo de análise de drift e impacto de features.

Exporta reports/drift_reports.csv com métricas mensais
(Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, KS, PSI médio)
e disponibiliza funções de feature importance para o modelo.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)


def _extract_feature_names(column_transformer):
    """Extrai nomes das features de um ColumnTransformer fitted."""
    names = []
    for name, trans, cols in column_transformer.transformers_:
        if name == 'remainder':
            continue
        if not isinstance(cols, list):
            cols = list(cols)

        # Tenta get_feature_names_out do pipeline/transformer
        try:
            feat_out = trans.get_feature_names_out(cols)
            names.extend(feat_out)
            continue
        except (AttributeError, ValueError):
            pass

        # Fallback: OneHotEncoder dentro de Pipeline
        pipeline_steps = dict(trans.steps) if hasattr(trans, 'steps') else {}
        encoder = pipeline_steps.get('encoder', None)
        if encoder is not None and hasattr(encoder, 'categories_'):
            for i, col in enumerate(cols):
                cats = encoder.categories_[i][1:]  # drop='first'
                for cat in cats:
                    names.append(f'{col}_{cat}')
        else:
            names.extend(cols)

    return names


def _calcular_psi(esperado, atual, bins=10):
    esperado = esperado.dropna()
    atual = atual.dropna()
    limites = np.unique(np.percentile(esperado, np.linspace(0, 100, bins + 1)))
    if len(limites) < 2:
        return 0.0
    limites[0] = -np.inf
    limites[-1] = np.inf
    cont_esperado, _ = np.histogram(esperado, bins=limites)
    cont_atual, _ = np.histogram(atual, bins=limites)
    cont_esperado = np.where(cont_esperado == 0, 0.001, cont_esperado)
    cont_atual = np.where(cont_atual == 0, 0.001, cont_atual)
    perc_esperado = cont_esperado / cont_esperado.sum()
    perc_atual = cont_atual / cont_atual.sum()
    return float(np.sum((perc_atual - perc_esperado) * np.log(perc_atual / perc_esperado)))


def _metricas_binarias(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    ks_val, _ = ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0])
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'KS': ks_val,
    }


def relatorio_drift_mensal(
    modelo,
    X_hist,
    y_hist,
    df_oot,
    target_col='RainTomorrow',
    threshold=0.5,
    col_mes='Month',
    bins_psi=10,
):
    """
    Gera um DataFrame com métricas de performance e PSI médio por mês do OOT.

    Colunas retornadas:
        Periodo, Accuracy, Precision, Recall, F1-Score,
        ROC-AUC, PR-AUC, KS, PSI_Medio, Taxa_Rain

    Parameters
    ----------
    modelo : estimator
        Pipeline sklearn já treinado.
    X_hist : pd.DataFrame
        Features do conjunto histórico.
    y_hist : pd.Series
        Target do conjunto histórico.
    df_oot : pd.DataFrame
        DataFrame OOT completo (com target e col_mes).
    target_col : str
        Nome da coluna alvo.
    threshold : float
        Limiar de decisão.
    col_mes : str
        Coluna com o número do mês.
    bins_psi : int
        Quantidade de bins para o PSI.

    Returns
    -------
    pd.DataFrame
    """
    cols_num = X_hist.select_dtypes(include=['float64', 'int64']).columns
    linhas = []

    for mes in sorted(df_oot[col_mes].unique()):
        df_mes = df_oot[df_oot[col_mes] == mes]
        if len(df_mes) == 0:
            continue

        X_mes = df_mes.drop(columns=[target_col])
        y_mes = df_mes[target_col]
        prob_mes = modelo.predict_proba(X_mes)[:, 1]

        metricas = _metricas_binarias(y_mes, prob_mes, threshold)

        psi_vals = []
        for col in cols_num:
            if col in X_mes.columns:
                psi_vals.append(
                    _calcular_psi(X_hist[col], X_mes[col], bins=bins_psi)
                )

        metricas['Periodo'] = f'Mês {mes:02d}'
        metricas['PSI_Medio'] = float(np.mean(psi_vals)) if psi_vals else np.nan
        metricas['Taxa_Rain'] = float(y_mes.mean())
        metricas['N_Amostras'] = len(df_mes)
        linhas.append(metricas)

    colunas = [
        'Periodo', 'Accuracy', 'Precision', 'Recall', 'F1-Score',
        'ROC-AUC', 'PR-AUC', 'KS', 'PSI_Medio', 'Taxa_Rain', 'N_Amostras',
    ]
    return pd.DataFrame(linhas)[colunas]


def exportar_drift_report(
    df_drift_mensal,
    pasta_reports=None,
    nome_arquivo='drift_reports.csv',
):
    """
    Exporta o relatório de drift mensal para reports/drift_reports.csv.

    Parameters
    ----------
    df_drift_mensal : pd.DataFrame
        DataFrame retornado por ``relatorio_drift_mensal``.
    pasta_reports : str, optional
        Caminho da pasta. Se None, usa ``reports/`` na raiz do projeto.
    nome_arquivo : str
        Nome do arquivo CSV de saída.
    """
    if pasta_reports is None:
        pasta_reports = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'reports',
        )

    os.makedirs(pasta_reports, exist_ok=True)
    caminho = os.path.join(pasta_reports, nome_arquivo)
    df_drift_mensal.to_csv(caminho, index=False)
    print(f"  drift_reports.csv -> {caminho}")
    return caminho


def feature_importance_pipeline(modelo, feature_names=None, top_n=20):
    """
    Extrai a importância das features de um modelo dentro de uma Pipeline.

    Para LogisticRegression: usa |coeficientes|.
    Para modelos baseados em árvore: usa feature_importances_.

    Parameters
    ----------
    modelo : Pipeline ou estimator
        Pipeline sklearn (com etapa 'clf') ou estimator direto.
    feature_names : list, optional
        Nomes das features após o preprocessor. Se None, tenta extrair
        automaticamente do ColumnTransformer.
    top_n : int
        Quantidade de features a retornar.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas ['Feature', 'Importancia', 'Direcao'],
        ordenado por importância decrescente.
    """
    if hasattr(modelo, 'named_steps'):
        clf = modelo.named_steps.get('clf', modelo)
        prep = modelo.named_steps.get('prep', None)
    else:
        clf = modelo
        prep = None

    if feature_names is None and prep is not None:
        feature_names = _extract_feature_names(prep)

    if hasattr(clf, 'coef_'):
        coefs = clf.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        importancias = np.abs(coefs)
        direcoes = np.where(coefs >= 0, '+', '-')
    elif hasattr(clf, 'feature_importances_'):
        importancias = clf.feature_importances_
        direcoes = np.full(len(importancias), '')
    else:
        raise ValueError("Modelo não possui coef_ nem feature_importances_.")

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importancias))]

    df = pd.DataFrame({
        'Feature': feature_names,
        'Importancia': importancias,
        'Direcao': direcoes,
    })
    return df.sort_values('Importancia', ascending=False).head(top_n).reset_index(drop=True)
