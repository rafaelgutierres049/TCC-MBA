"""
Agente 5 — Seleção e Treinamento
Spec: seção 6.5

Responsabilidade: Comparar algoritmos candidatos e treinar o mais adequado.

Candidatos por tipo (spec 6.5):
  classification:     Logistic Regression, Random Forest, XGBoost, SVM
  regression:         Linear Regression, Random Forest, XGBoost, SVR
  clustering:         K-Means, DBSCAN, Agglomerative
  time_series:        tratado como regressão com features temporais
  anomaly_detection:  Isolation Forest, LOF, One-Class SVM

Métricas padrão:
  classification   → F1-score weighted (cross-validation 5-fold estratificado)
  regression       → RMSE (cross-validation 5-fold)
  clustering       → Silhouette score
  anomaly_detection → decision_function score médio

Invariantes (spec 6.5):
  - Apenas o modelo vencedor é registrado no State
  - justification é obrigatório
  - Se nenhum candidato atingir desempenho mínimo → curto-circuito
  - Edge case 6: clustering → não usa coluna target

Desempenho mínimo aceitável:
  classification   → F1 >= 0.30 (acima de classificador aleatório para a maioria dos casos)
  regression       → R² >= 0.0 (melhor que prever a média)
  clustering       → Silhouette >= -0.5 (sempre passa exceto divergência completa)
  anomaly_detection → sempre passa (sem supervised threshold)
"""

import io
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR, OneClassSVM

from core.state import AgenticMLState

AGENT_NAME = "Agente 5 — Seleção e Treinamento"
logger = logging.getLogger(__name__)

_MIN_ACCEPTABLE: dict[str, float] = {
    "classification": 0.30,
    "regression": 0.0,       # R²
    "clustering": -0.5,
    "time_series": 0.0,
    "anomaly_detection": -999.0,  # sempre passa
}

_DEFAULT_METRIC: dict[str, str] = {
    "classification": "f1_weighted",
    "regression": "r2",          # R² como critério de seleção; RMSE reportado também
    "clustering": "silhouette",
    "time_series": "r2",
    "anomaly_detection": "decision_score",
}


# ─── Entry point do nó LangGraph ─────────────────────────────────────────────

def run_agent5(state: AgenticMLState) -> dict:
    """Nó LangGraph do Agente 5."""
    features: dict = state["features"] or {}
    interpretation: dict = state["interpretation"] or {}

    df_transformed: pd.DataFrame = features.get("dataset_transformed")
    target_col: str | None = features.get("target_column")
    problem_type: str = interpretation.get("problem_type", "classification")

    if df_transformed is None or df_transformed.empty:
        return _error("Dataset transformado está vazio — Agente 3 pode não ter concluído corretamente.")

    # Prepara X e y
    try:
        X, y, label_enc = _prepare_data(df_transformed, target_col, problem_type)
    except Exception as exc:
        return _error(f"Falha ao preparar dados para treinamento: {exc}")

    if X.shape[0] < 10:
        return _error(
            f"O dataset possui apenas {X.shape[0]} amostras após o pré-processamento, "
            "o que é insuficiente para treinar e validar modelos."
        )

    # Executa a competição entre candidatos
    try:
        results, metric_name = _evaluate_candidates(X, y, problem_type)
    except Exception as exc:
        return _error(f"Falha durante a avaliação dos modelos: {exc}")

    if not results:
        return _error("Nenhum modelo candidato conseguiu ser avaliado com os dados fornecidos.")

    # Seleciona o vencedor
    best = max(results, key=lambda r: r["score"])

    # Verifica desempenho mínimo aceitável (spec 6.5)
    min_score = _MIN_ACCEPTABLE.get(problem_type, 0.0)
    if best["score"] < min_score:
        return _error(
            f"Nenhum modelo atingiu desempenho mínimo aceitável. "
            f"Melhor resultado: {best['model']} com {metric_name}={best['score']:.4f} "
            f"(mínimo exigido: {min_score:.2f}). "
            "Verifique a qualidade do dataset e os issues reportados pelo Agente 2."
        )

    # Treina o modelo vencedor no dataset completo
    try:
        model_bytes = _train_final(X, y, best["estimator"])
    except Exception as exc:
        return _error(f"Falha ao treinar o modelo final: {exc}")

    justification = _build_justification(best, results, metric_name, problem_type)

    # Computa RMSE final para regressão (spec menciona RMSE como métrica de regressão)
    rmse_value: float | None = None
    if problem_type in ("regression", "time_series") and y is not None:
        best["estimator"].fit(X.values, y.values)
        y_pred = best["estimator"].predict(X.values)
        rmse_value = round(float(mean_squared_error(y.values, y_pred) ** 0.5), 4)

    logger.info(
        "[Agente 5] Modelo selecionado: %s | %s=%.4f",
        best["model"], metric_name, best["score"],
    )

    training_output: dict = {
        "model_selected": best["model"],
        "metric_used": metric_name,
        "metric_value": round(best["score"], 4),
        "justification": justification,
        "model_artifact": model_bytes,
        "candidate_results": [
            {"model": r["model"], "score": round(r["score"], 4)} for r in results
        ],
        "label_encoder_target": label_enc,
    }
    if rmse_value is not None:
        training_output["rmse"] = rmse_value

    return {"training": training_output, "status": "running"}


# ─── Preparação dos dados ─────────────────────────────────────────────────────

def _prepare_data(
    df: pd.DataFrame, target_col: str | None, problem_type: str
) -> tuple[pd.DataFrame, pd.Series | None, LabelEncoder | None]:
    """
    Separa X e y, garante que X é totalmente numérico.
    Aplica LabelEncoder no target se for categórico.
    """
    # Clustering e anomaly detection: sem target supervisionado
    if problem_type in ("clustering", "anomaly_detection") or target_col is None:
        X = df.copy()
        X = _to_numeric(X)
        return X, None, None

    if target_col not in df.columns:
        raise ValueError(f"Coluna target '{target_col}' não encontrada no dataset transformado.")

    y_raw = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    X = _to_numeric(X)

    label_enc: LabelEncoder | None = None
    if not pd.api.types.is_numeric_dtype(y_raw):
        label_enc = LabelEncoder()
        y = pd.Series(label_enc.fit_transform(y_raw.astype(str)), name=target_col)
    else:
        y = y_raw.dropna()
        X = X.loc[y.index]

    return X, y, label_enc


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que todas as colunas de X são numéricas.
    Converte bool → int, object → LabelEncoder, preenche NaN residuais com 0.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
        elif not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df.fillna(0)


# ─── Candidatos por tipo de problema ─────────────────────────────────────────

def _get_candidates(problem_type: str) -> list[tuple[str, Any]]:
    try:
        from xgboost import XGBClassifier, XGBRegressor
        has_xgb = True
    except Exception:
        has_xgb = False

    if problem_type == "classification":
        candidates = [
            ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("SVM", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
        if has_xgb:
            candidates.insert(
                2,
                ("XGBoost", XGBClassifier(
                    eval_metric="logloss", random_state=42,
                    verbosity=0, use_label_encoder=False,
                )),
            )
        return candidates

    if problem_type in ("regression", "time_series"):
        candidates = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("SVR", SVR(kernel="rbf")),
        ]
        if has_xgb:
            from xgboost import XGBRegressor
            candidates.insert(2, ("XGBoost", XGBRegressor(random_state=42, verbosity=0)))
        return candidates

    if problem_type == "clustering":
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        return [
            ("K-Means", KMeans(n_clusters=3, random_state=42, n_init=10)),
            ("DBSCAN", DBSCAN(eps=0.5, min_samples=5)),
            ("Agglomerative", AgglomerativeClustering(n_clusters=3)),
        ]

    if problem_type == "anomaly_detection":
        return [
            ("Isolation Forest", IsolationForest(random_state=42, contamination=0.1)),
            ("LOF", LocalOutlierFactor(novelty=True, contamination=0.1)),
            ("One-Class SVM", OneClassSVM(nu=0.1)),
        ]

    return []


# ─── Avaliação dos candidatos ─────────────────────────────────────────────────

def _evaluate_candidates(
    X: pd.DataFrame, y: pd.Series | None, problem_type: str
) -> tuple[list[dict], str]:
    candidates = _get_candidates(problem_type)
    metric_name = _DEFAULT_METRIC.get(problem_type, "score")
    n_samples = X.shape[0]
    n_folds = min(5, max(2, n_samples // 10))

    results: list[dict] = []

    for name, estimator in candidates:
        try:
            score = _score_candidate(
                estimator, X, y, problem_type, n_folds, metric_name
            )
            results.append({"model": name, "score": score, "estimator": estimator})
            logger.info("[Agente 5] %s → %s=%.4f", name, metric_name, score)
        except Exception as exc:
            logger.warning("[Agente 5] %s falhou: %s", name, exc)

    return results, metric_name


def _score_candidate(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | None,
    problem_type: str,
    n_folds: int,
    metric_name: str,
) -> float:

    X_arr = X.values

    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            estimator, X_arr, y.values, cv=cv, scoring="f1_weighted"
        )
        return float(scores.mean())

    if problem_type in ("regression", "time_series"):
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        r2_scores = cross_val_score(estimator, X_arr, y.values, cv=cv, scoring="r2")
        return float(r2_scores.mean())

    if problem_type == "clustering":
        estimator.fit(X_arr)
        if hasattr(estimator, "labels_"):
            labels = estimator.labels_
        else:
            labels = estimator.fit_predict(X_arr)
        n_unique = len(set(labels)) - (1 if -1 in labels else 0)
        if n_unique < 2:
            return -1.0
        return float(silhouette_score(X_arr, labels))

    if problem_type == "anomaly_detection":
        estimator.fit(X_arr)
        scores = estimator.decision_function(X_arr)
        return float(scores.mean())

    return -999.0


# ─── Treinamento final e serialização ────────────────────────────────────────

def _train_final(X: pd.DataFrame, y: pd.Series | None, estimator: Any) -> bytes:
    """Treina o modelo vencedor no dataset completo e serializa com joblib."""
    X_arr = X.values
    if y is not None:
        estimator.fit(X_arr, y.values)
    else:
        estimator.fit(X_arr)

    buf = io.BytesIO()
    joblib.dump(estimator, buf)
    return buf.getvalue()


# ─── Justificativa ───────────────────────────────────────────────────────────

def _build_justification(
    best: dict, results: list[dict], metric_name: str, problem_type: str
) -> str:
    others = [r for r in results if r["model"] != best["model"]]
    comparison = ", ".join(
        f"{r['model']} ({r['score']:.4f})" for r in sorted(others, key=lambda r: -r["score"])
    )
    metric_label = {
        "f1_weighted": "F1-score ponderado",
        "rmse": "R² (proxy de RMSE)",
        "silhouette": "Silhouette score",
        "decision_score": "decision function",
    }.get(metric_name, metric_name)

    justification = (
        f"{best['model']} foi selecionado com {metric_label} de {best['score']:.4f} "
        f"para o problema de {problem_type}."
    )
    if comparison:
        justification += f" Demais candidatos avaliados: {comparison}."
    return justification


# ─── Error helper ─────────────────────────────────────────────────────────────

def _error(message: str) -> dict:
    return {
        "status": "error",
        "failed_at": AGENT_NAME,
        "error": {
            "technical": message,
            "accessible": (
                "O sistema não conseguiu treinar um modelo adequado para os seus dados. "
                f"Detalhe: {message}"
            ),
        },
    }
