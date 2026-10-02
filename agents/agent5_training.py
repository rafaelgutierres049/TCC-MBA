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
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
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
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR, OneClassSVM

from core.state import AgenticMLState

AGENT_NAME = "Agente 5 — Seleção e Treinamento"
logger = logging.getLogger(__name__)

# Busca leve de hiperparâmetros (RandomizedSearchCV) dentro de cada candidato da spec 6.5.
# Sem isso, o Agente 5 competia com hiperparâmetros fixos/default contra baselines (ex. FLAML)
# que otimizam hiperparâmetros — não é a mesma disputa. Mantém os MESMOS candidatos da spec,
# só passa a buscar dentro do espaço de cada um. Clustering/anomaly_detection ficam de fora
# (scoring não-supervisionado não é compatível com RandomizedSearchCV sem trabalho adicional).
_SEARCH_ITER = 10

_PARAM_DISTRIBUTIONS: dict[str, dict] = {
    "Logistic Regression": {"C": loguniform(1e-2, 1e2)},
    "Random Forest": {
        "n_estimators": randint(100, 400),
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": randint(1, 5),
    },
    "XGBoost": {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 8),
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
    },
    "SVM": {"C": loguniform(1e-1, 1e2), "gamma": loguniform(1e-3, 1.0)},
    "SVR": {"C": loguniform(1e-1, 1e2), "gamma": loguniform(1e-3, 1.0), "epsilon": loguniform(1e-3, 1.0)},
}


def _clean_params(params: dict) -> dict:
    """Converte escalares numpy (np.float64/np.int64) para tipos nativos — legibilidade
    na justificativa e no relatório do Agente 6, e serialização JSON sem surpresas."""
    cleaned = {}
    for k, v in params.items():
        if isinstance(v, (np.floating,)):
            cleaned[k] = round(float(v), 4)
        elif isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        else:
            cleaned[k] = v
    return cleaned

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
    seed: int = state.get("random_state", 42)
    balance_strategy: str = state.get("balance_strategy", "auto")

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

    # Decide o tratamento de desbalanceamento (spec 6.2 recomenda balanceamento; aplicado aqui)
    balancing = _resolve_balancing(balance_strategy, problem_type, y)

    # Executa a competição entre candidatos
    try:
        results, metric_name = _evaluate_candidates(
            X, y, problem_type, seed, balanced=balancing["applied"]
        )
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

    justification = _build_justification(best, results, metric_name, problem_type, balancing)

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
        "best_params": best.get("best_params"),
        "label_encoder_target": label_enc,
        "balancing": balancing,
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

# Detecção (Agente 2) começa em 60%; ação automática só a partir de 80% —
# desbalanceamento moderado é reportado mas não altera o pipeline sem opt-in.
_IMBALANCE_ACTION_THRESHOLD = 0.80


def _resolve_balancing(strategy: str, problem_type: str, y: pd.Series | None) -> dict:
    """
    Decide o tratamento de desbalanceamento com base na distribuição real do target.
      strategy="none"          → nunca aplica
      strategy="class_weight"  → sempre aplica (para o A/B do TCC)
      strategy="auto" (padrão) → aplica se a classe dominante ≥ _IMBALANCE_ACTION_THRESHOLD (80%)
    """
    if problem_type != "classification" or y is None:
        return {"strategy": strategy, "applied": False, "method": "nenhum",
                "reason": "não se aplica (problema não é de classificação)"}

    dist = y.value_counts(normalize=True)
    dominant = float(dist.iloc[0]) if len(dist) else 0.0
    dominant_pct = round(dominant * 100, 2)
    limiar_pct = f"{_IMBALANCE_ACTION_THRESHOLD * 100:.0f}%"

    if strategy == "none":
        return {"strategy": "none", "applied": False, "method": "nenhum",
                "dominant_class_pct": dominant_pct,
                "reason": "desativado explicitamente (balance_strategy='none')"}
    if strategy == "class_weight":
        return {"strategy": "class_weight", "applied": True, "method": "class_weight=balanced",
                "dominant_class_pct": dominant_pct,
                "reason": "forçado explicitamente (balance_strategy='class_weight')"}

    applied = dominant >= _IMBALANCE_ACTION_THRESHOLD
    return {
        "strategy": "auto",
        "applied": applied,
        "method": "class_weight=balanced" if applied else "nenhum",
        "dominant_class_pct": dominant_pct,
        "reason": (
            f"classe dominante em {dominant_pct}% (≥ limiar de ação {limiar_pct}) → class_weight='balanced' aplicado"
            if applied else
            f"classe dominante em {dominant_pct}% (< limiar de ação {limiar_pct}) → sem balanceamento automático"
        ),
    }


def _get_candidates(
    problem_type: str, seed: int = 42, balanced: bool = False
) -> list[tuple[str, Any]]:
    try:
        from xgboost import XGBClassifier, XGBRegressor
        has_xgb = True
    except Exception:
        has_xgb = False

    cw = "balanced" if balanced else None

    if problem_type == "classification":
        candidates = [
            ("Logistic Regression", LogisticRegression(max_iter=500, random_state=seed, class_weight=cw)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=seed, class_weight=cw)),
            ("SVM", SVC(kernel="rbf", probability=True, random_state=seed, class_weight=cw)),
        ]
        # XGBoost não usa class_weight; o balanceamento de classes fica a cargo dos demais candidatos.
        if has_xgb:
            candidates.insert(
                2,
                ("XGBoost", XGBClassifier(
                    eval_metric="logloss", random_state=seed,
                    verbosity=0, use_label_encoder=False,
                )),
            )
        return candidates

    if problem_type in ("regression", "time_series"):
        candidates = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=seed)),
            ("SVR", SVR(kernel="rbf")),
        ]
        if has_xgb:
            from xgboost import XGBRegressor
            candidates.insert(2, ("XGBoost", XGBRegressor(random_state=seed, verbosity=0)))
        return candidates

    if problem_type == "clustering":
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        return [
            ("K-Means", KMeans(n_clusters=3, random_state=seed, n_init=10)),
            ("DBSCAN", DBSCAN(eps=0.5, min_samples=5)),
            ("Agglomerative", AgglomerativeClustering(n_clusters=3)),
        ]

    if problem_type == "anomaly_detection":
        return [
            ("Isolation Forest", IsolationForest(random_state=seed, contamination=0.1)),
            ("LOF", LocalOutlierFactor(novelty=True, contamination=0.1)),
            ("One-Class SVM", OneClassSVM(nu=0.1)),
        ]

    return []


# ─── Avaliação dos candidatos ─────────────────────────────────────────────────

def _evaluate_candidates(
    X: pd.DataFrame, y: pd.Series | None, problem_type: str, seed: int = 42,
    balanced: bool = False,
) -> tuple[list[dict], str]:
    candidates = _get_candidates(problem_type, seed, balanced=balanced)
    metric_name = _DEFAULT_METRIC.get(problem_type, "score")
    n_samples = X.shape[0]
    n_folds = min(5, max(2, n_samples // 10))

    results: list[dict] = []

    for name, estimator in candidates:
        try:
            score, tuned_estimator, best_params = _score_candidate(
                name, estimator, X, y, problem_type, n_folds, metric_name, seed
            )
            results.append({
                "model": name, "score": score,
                "estimator": tuned_estimator, "best_params": best_params,
            })
            logger.info("[Agente 5] %s → %s=%.4f%s", name, metric_name, score,
                        f" | params={best_params}" if best_params else "")
        except Exception as exc:
            logger.warning("[Agente 5] %s falhou: %s", name, exc)

    return results, metric_name


def _score_candidate(
    name: str,
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | None,
    problem_type: str,
    n_folds: int,
    metric_name: str,
    seed: int = 42,
) -> tuple[float, Any, dict | None]:
    """
    Avalia um candidato via cross-validation.
    Para classification/regression, quando há espaço de busca em _PARAM_DISTRIBUTIONS,
    faz RandomizedSearchCV (tuning leve dentro do mesmo esquema de CV) em vez de usar
    apenas os hiperparâmetros default — dá ao Agente 5 uma chance real de competir com
    baselines de AutoML que otimizam hiperparâmetros.
    Retorna (score, estimador pronto para o treino final, melhores hiperparâmetros ou None).
    """
    X_arr = X.values
    dist = _PARAM_DISTRIBUTIONS.get(name)

    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        if dist:
            search = RandomizedSearchCV(
                estimator, dist, n_iter=_SEARCH_ITER, cv=cv,
                scoring="f1_weighted", random_state=seed, n_jobs=-1,
            )
            search.fit(X_arr, y.values)
            return float(search.best_score_), clone(search.best_estimator_), _clean_params(search.best_params_)
        scores = cross_val_score(estimator, X_arr, y.values, cv=cv, scoring="f1_weighted")
        return float(scores.mean()), estimator, None

    if problem_type in ("regression", "time_series"):
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        if dist:
            search = RandomizedSearchCV(
                estimator, dist, n_iter=_SEARCH_ITER, cv=cv,
                scoring="r2", random_state=seed, n_jobs=-1,
            )
            search.fit(X_arr, y.values)
            return float(search.best_score_), clone(search.best_estimator_), _clean_params(search.best_params_)
        r2_scores = cross_val_score(estimator, X_arr, y.values, cv=cv, scoring="r2")
        return float(r2_scores.mean()), estimator, None

    if problem_type == "clustering":
        estimator.fit(X_arr)
        if hasattr(estimator, "labels_"):
            labels = estimator.labels_
        else:
            labels = estimator.fit_predict(X_arr)
        n_unique = len(set(labels)) - (1 if -1 in labels else 0)
        if n_unique < 2:
            return -1.0, estimator, None
        return float(silhouette_score(X_arr, labels)), estimator, None

    if problem_type == "anomaly_detection":
        estimator.fit(X_arr)
        scores = estimator.decision_function(X_arr)
        return float(scores.mean()), estimator, None

    return -999.0, estimator, None


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
    best: dict, results: list[dict], metric_name: str, problem_type: str,
    balancing: dict | None = None,
) -> str:
    others = [r for r in results if r["model"] != best["model"]]
    comparison = ", ".join(
        f"{r['model']} ({r['score']:.4f})" for r in sorted(others, key=lambda r: -r["score"])
    )
    metric_label = {
        "f1_weighted": "F1-score ponderado",
        "r2": "R²",
        "silhouette": "Silhouette score",
        "decision_score": "decision function",
    }.get(metric_name, metric_name)

    justification = (
        f"{best['model']} foi selecionado com {metric_label} de {best['score']:.4f} "
        f"para o problema de {problem_type}."
    )
    if comparison:
        justification += f" Demais candidatos avaliados: {comparison}."
    if best.get("best_params"):
        justification += f" Hiperparâmetros otimizados via busca aleatória (CV): {best['best_params']}."
    if balancing and balancing.get("applied"):
        justification += (
            f" Tratamento de desbalanceamento: {balancing['method']} "
            f"({balancing.get('reason', '')})."
        )
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
