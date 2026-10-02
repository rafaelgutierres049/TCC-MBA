"""
Harness de avaliação do TCC — protocolo com holdout isolado.

Motivação (parecer da orientação):
  - A métrica reportada pelo AgenticML (`training.metric_value`) é a média da
    validação cruzada usada para *selecionar* o modelo. Usar esse mesmo número
    como métrica final gera viés otimista.
  - A comparação com o baseline (FLAML) precisa usar EXATAMENTE a mesma partição
    train/test, com um conjunto de teste nunca visto durante a seleção.

Este módulo NÃO faz parte do sistema AgenticML (spec) — é scaffolding de avaliação
para os notebooks do TCC.

Uso típico:
    from tcc_eval import make_split, evaluate_agenticml, evaluate_flaml, export_split

    train_df, test_df, info = make_split(df, target="species", task="classification", seed=42)
    ag = evaluate_agenticml(train_df, test_df, target="species",
                            prompt="Classificar a espécie da flor de íris...",
                            hint="classification", seed=42)
    print(ag["holdout_metric_name"], ag["holdout_metric"])   # métrica honesta no holdout
    print("gap de otimismo:", ag["cv_score"] - ag["holdout_metric"])
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

import io

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# Garante que a raiz do projeto está no path (para importar core/ e agents/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.agent6_explanation import (  # noqa: E402
    _align_features,
    _apply_transformations,
    _generate_predictions,
    _SchemaError,
)
from core.graph import build_graph  # noqa: E402
from core.state import initial_state  # noqa: E402

CLASSIFICATION = "classification"
REGRESSION = "regression"


# ─── Registro central de datasets do TCC ───────────────────────────────────
# Evita duplicar o código de carga entre os notebooks 01/02/03/05.

def _load_iris_df() -> pd.DataFrame:
    from sklearn.datasets import load_iris

    it = load_iris(as_frame=True)
    d = it.frame.copy()
    d.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in d.columns]
    d["species"] = it.target_names[d["target"]]
    return d.drop(columns=["target"])


def _load_titanic_df() -> pd.DataFrame:
    import seaborn as sns

    cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]
    return sns.load_dataset("titanic")[cols].copy()


def _load_california_df(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing

    frame = fetch_california_housing(as_frame=True).frame
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


DATASETS: dict[str, dict] = {
    "Iris": {
        "loader": _load_iris_df,
        "target": "species",
        "task": CLASSIFICATION,
        "hint": "classification",
        "prompt": (
            "Classificar a espécie da flor de íris (setosa, versicolor ou virginica) "
            "com base nas medidas das pétalas e sépalas."
        ),
    },
    "Titanic": {
        "loader": _load_titanic_df,
        "target": "survived",
        "task": CLASSIFICATION,
        "hint": "classification",
        "prompt": (
            "Prever se um passageiro sobreviveu ao naufrágio do Titanic com base "
            "nas características socioeconômicas e demográficas."
        ),
    },
    "California Housing": {
        "loader": _load_california_df,
        "target": "MedHouseVal",
        "task": REGRESSION,
        "hint": "regression",
        "prompt": (
            "Estimar o valor mediano das casas (em centenas de milhares de dólares) "
            "com base nas características demográficas e geográficas do bloco censitário."
        ),
    },
}


def load_dataset(name: str) -> tuple[pd.DataFrame, str, str, str, str | None]:
    """Retorna (df, target, task, prompt, hint) para um dataset do registro."""
    if name not in DATASETS:
        raise KeyError(f"Dataset '{name}' não registrado. Opções: {list(DATASETS)}")
    cfg = DATASETS[name]
    return cfg["loader"](), cfg["target"], cfg["task"], cfg["prompt"], cfg["hint"]


# ─── Split reprodutível ──────────────────────────────────────────────────────

def make_split(
    df: pd.DataFrame,
    target: str,
    task: str,
    test_size: float = 0.2,
    seed: int = 42,
    reg_stratify_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Divide o dataset em train/test.

    - classification: estratificado pela própria target.
    - regression: estratificado por quantis da target (estabiliza a partição
      entre sementes diferentes).

    Retorna (train_df, test_df, split_info). `split_info` registra a semente e os
    índices originais do conjunto de teste — suficiente para reconstruir a
    partição idêntica em outro ambiente (ex.: Colab, para o FLAML).
    """
    if target not in df.columns:
        raise KeyError(f"Coluna target '{target}' ausente no DataFrame.")

    df = df.reset_index(drop=True)
    y = df[target]

    strat: pd.Series | None = None
    if task == CLASSIFICATION:
        strat = y
    elif reg_stratify_bins and y.nunique() > reg_stratify_bins:
        strat = pd.qcut(y, q=reg_stratify_bins, duplicates="drop")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True, stratify=strat
    )

    info = {
        "task": task,
        "target": target,
        "seed": seed,
        "test_size": test_size,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_index": test_df.index.tolist(),
        "stratified": strat is not None,
    }
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), info


def export_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
    name: str,
) -> dict:
    """
    Grava train/test em CSV para que o mesmo split seja consumido por outro
    ambiente (ex.: FLAML no Colab). Retorna os caminhos gravados.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / f"{name}_train.csv"
    test_path = out / f"{name}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return {"train": str(train_path), "test": str(test_path)}


# ─── Métrica no holdout ─────────────────────────────────────────────────────

def score_holdout(
    y_true: Any, y_pred: Any, task: str
) -> tuple[str, float, float | None]:
    """
    classification → ('f1_weighted', valor, None)
    regression     → ('r2', valor, rmse)
    Alinha dtypes entre y_true e y_pred antes de comparar.
    """
    y_true = pd.Series(list(y_true)).reset_index(drop=True)
    y_pred = pd.Series(list(y_pred)).reset_index(drop=True)

    if task == CLASSIFICATION:
        if pd.api.types.is_numeric_dtype(y_true):
            y_pred = pd.to_numeric(y_pred, errors="coerce").round().astype(y_true.dtype)
        else:
            y_true = y_true.astype(str)
            y_pred = y_pred.astype(str)
        return "f1_weighted", float(f1_score(y_true, y_pred, average="weighted")), None

    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    return (
        "r2",
        float(r2_score(y_true, y_pred)),
        float(root_mean_squared_error(y_true, y_pred)),
    )


# ─── AgenticML no holdout ───────────────────────────────────────────────────

def evaluate_agenticml(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    prompt: str,
    hint: str | None,
    seed: int = 42,
    balance_strategy: str = "auto",
    return_state: bool = False,
) -> dict:
    """
    Treina o AgenticML SOMENTE no train_df e avalia no test_df (holdout isolado).
    A previsão no holdout usa o mesmo caminho do endpoint /predict
    (_generate_predictions), garantindo consistência com o produto.

    `balance_strategy` ("auto" | "none" | "class_weight") permite o A/B de
    desbalanceamento pedido no parecer (impacto no F1 ponderado).

    `return_state=True` inclui o `final_state` completo do grafo em `"_final_state"`
    — usado pelos notebooks para exibir as seções narrativas por agente sem
    rodar o pipeline duas vezes (uma para narrativa, outra para o holdout).
    """
    task = REGRESSION if hint == REGRESSION else CLASSIFICATION

    state = initial_state(
        dataset_bytes=train_df.to_csv(index=False).encode(),
        filename="train.csv",
        prompt=prompt,
        problem_type_hint=hint,
        random_state=seed,
        balance_strategy=balance_strategy,
    )

    t0 = time.time()
    final = build_graph().invoke(state)
    elapsed = round(time.time() - t0, 2)

    out = final.get("output") or {}
    status = out.get("status", final.get("status"))
    if status != "done":
        result = {
            "status": status,
            "failed_at": final.get("failed_at") or out.get("failed_at"),
            "error": (out.get("explanation") or final.get("error") or {}).get("technical"),
            "elapsed": elapsed,
        }
        if return_state:
            result["_final_state"] = final
        return result

    features = final.get("features") or {}
    training = final.get("training") or {}
    chosen_target = features.get("target_column")
    problem_type = (final.get("interpretation") or {}).get("problem_type")
    balancing = training.get("balancing") or {}

    features_dict = {
        "fitted_params": features.get("fitted_params", {}),
        "feature_columns": features.get("feature_columns", []),
    }
    training_dict = {
        "model_artifact": training.get("model_artifact"),
        "label_encoder_target": training.get("label_encoder_target"),
    }

    try:
        preds = _generate_predictions(
            test_df.to_csv(index=False).encode(), "test.csv", features_dict, training_dict
        )
    except _SchemaError as exc:
        result = {"status": "schema_error", "error": str(exc), "elapsed": elapsed}
        if return_state:
            result["_final_state"] = final
        return result

    y_pred = preds["values"]
    y_true = test_df[target].tolist()
    metric_name, metric_value, rmse = score_holdout(y_true, y_pred, task)

    result = {
        "status": "done",
        "model_selected": training.get("model_selected"),
        "problem_type": problem_type,
        "target_declared": target,
        "target_chosen_by_agent": chosen_target,
        "target_mismatch": bool(chosen_target and chosen_target != target),
        "cv_score": training.get("metric_value"),         # score interno de SELEÇÃO (CV no train)
        "cv_metric": training.get("metric_used"),
        "balancing_applied": bool(balancing.get("applied")),
        "balancing_method": balancing.get("method"),
        "balancing_reason": balancing.get("reason"),
        "holdout_metric_name": metric_name,
        "holdout_metric": round(metric_value, 4),
        "holdout_rmse": round(rmse, 4) if rmse is not None else None,
        "optimism_gap": (
            round(training.get("metric_value") - metric_value, 4)
            if isinstance(training.get("metric_value"), (int, float))
            else None
        ),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "elapsed": elapsed,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if return_state:
        result["_final_state"] = final
    return result


# ─── FLAML no holdout (mesma partição) ──────────────────────────────────────

def evaluate_flaml(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    task: str,
    time_budget: int = 240,
    seed: int = 42,
    metric: str | None = None,
) -> dict:
    """
    Baseline: FLAML AutoML treinado no MESMO train_df e avaliado no MESMO test_df.
    Requer `pip install flaml[automl]`. Recebe os dados brutos (FLAML faz o próprio
    pré-processamento), espelhando o que o AgenticML recebe.
    """
    try:
        from flaml import AutoML
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FLAML não instalado. Rode `pip install 'flaml[automl]'` "
            "ou execute esta etapa no Colab consumindo os CSVs de export_split()."
        ) from exc

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    if metric is None:
        metric = "f1" if task == CLASSIFICATION else "r2"

    automl = AutoML()
    t0 = time.time()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=task,
        time_budget=time_budget,
        metric=metric,
        seed=seed,
        verbose=0,
    )
    elapsed = round(time.time() - t0, 2)

    y_pred = automl.predict(X_test)
    metric_name, metric_value, rmse = score_holdout(y_test, y_pred, task)

    return {
        "status": "done",
        "best_estimator": str(automl.best_estimator),
        "time_budget": time_budget,
        "holdout_metric_name": metric_name,
        "holdout_metric": round(metric_value, 4),
        "holdout_rmse": round(rmse, 4) if rmse is not None else None,
        "elapsed": elapsed,
        "y_true": y_test.tolist(),
        "y_pred": list(np.asarray(y_pred).tolist()),
    }


# ─── Repetições multi-seed ──────────────────────────────────────────────────

def run_multiseed(
    df: pd.DataFrame,
    dataset_name: str,
    target: str,
    prompt: str,
    hint: str | None,
    task: str,
    seeds: list[int],
    test_size: float = 0.2,
    balance_strategy: str = "auto",
    run_flaml: bool = False,
    flaml_time_budget: int = 240,
    on_progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Para cada semente: gera a MESMA partição para os dois sistemas e avalia no holdout.

    Retorna um DataFrame long-format (uma linha por (dataset, system, seed)):
        dataset | system | seed | metric_name | metric | rmse | model | elapsed
                | cv_score | optimism_gap    (cv_score/optimism_gap só p/ AgenticML)

    Se `run_flaml=False` (ou FLAML ausente), só as linhas do AgenticML são geradas;
    complete o FLAML depois com `merge_flaml_csv()` (fluxo Colab).
    """
    rows: list[dict] = []
    for seed in seeds:
        train_df, test_df, _ = make_split(df, target, task, test_size=test_size, seed=seed)

        ag = evaluate_agenticml(
            train_df, test_df, target, prompt, hint, seed=seed,
            balance_strategy=balance_strategy,
        )
        if ag.get("status") != "done":
            if on_progress:
                on_progress(f"[{dataset_name} seed={seed}] AgenticML FALHOU: {ag.get('error')}")
            continue
        rows.append({
            "dataset": dataset_name, "system": "AgenticML", "seed": seed,
            "metric_name": ag["holdout_metric_name"], "metric": ag["holdout_metric"],
            "rmse": ag["holdout_rmse"], "model": ag["model_selected"],
            "elapsed": ag["elapsed"], "cv_score": ag["cv_score"],
            "optimism_gap": ag["optimism_gap"],
            "balancing": ag["balancing_method"] if ag["balancing_applied"] else "nenhum",
        })
        if on_progress:
            on_progress(
                f"[{dataset_name} seed={seed}] AgenticML {ag['model_selected']} "
                f"{ag['holdout_metric_name']}={ag['holdout_metric']}"
            )

        if run_flaml:
            fl = evaluate_flaml(
                train_df, test_df, target, task,
                time_budget=flaml_time_budget, seed=seed,
            )
            rows.append({
                "dataset": dataset_name, "system": "FLAML", "seed": seed,
                "metric_name": fl["holdout_metric_name"], "metric": fl["holdout_metric"],
                "rmse": fl["holdout_rmse"], "model": fl["best_estimator"],
                "elapsed": fl["elapsed"], "cv_score": None, "optimism_gap": None,
                "balancing": None,
            })
            if on_progress:
                on_progress(
                    f"[{dataset_name} seed={seed}] FLAML {fl['best_estimator']} "
                    f"{fl['holdout_metric_name']}={fl['holdout_metric']}"
                )

    return pd.DataFrame(rows)


def merge_flaml_csv(long_df: pd.DataFrame, csv_path: str | Path) -> pd.DataFrame:
    """
    Fluxo Colab: anexa resultados do FLAML rodados em outro ambiente.
    O CSV deve ter colunas: dataset, seed, metric_name, metric[, rmse, model, elapsed].
    """
    fl = pd.read_csv(csv_path)
    fl["system"] = "FLAML"
    for col in ("rmse", "model", "elapsed", "cv_score", "optimism_gap", "balancing"):
        if col not in fl.columns:
            fl[col] = None
    return pd.concat([long_df, fl[long_df.columns]], ignore_index=True)


# ─── Agregação e testes estatísticos ───────────────────────────────────────

def ci95(values: Any) -> tuple[float, float]:
    """Intervalo de confiança 95% da média (t de Student). n<2 → (nan, nan)."""
    from scipy import stats

    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = a.mean()
    sem = stats.sem(a)
    h = sem * stats.t.ppf(0.975, n - 1)
    return (float(mean - h), float(mean + h))


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uma linha por (dataset, system): n, média, desvio, IC95%, min, max da métrica
    de holdout. Também agrega o RMSE quando presente.
    """
    out: list[dict] = []
    for (dataset, system), g in long_df.groupby(["dataset", "system"], sort=False):
        vals = g["metric"].to_numpy(dtype=float)
        lo, hi = ci95(vals)
        row = {
            "dataset": dataset,
            "system": system,
            "metric_name": g["metric_name"].iloc[0],
            "n": len(vals),
            "mean": round(float(np.nanmean(vals)), 4),
            "std": round(float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
            "min": round(float(np.nanmin(vals)), 4),
            "max": round(float(np.nanmax(vals)), 4),
        }
        if g["rmse"].notna().any():
            rvals = g["rmse"].to_numpy(dtype=float)
            row["rmse_mean"] = round(float(np.nanmean(rvals)), 4)
            row["rmse_std"] = round(float(np.nanstd(rvals, ddof=1)) if len(rvals) > 1 else 0.0, 4)
        out.append(row)
    return pd.DataFrame(out)


def paired_comparison(
    long_df: pd.DataFrame, dataset: str, alpha: float = 0.05
) -> dict:
    """
    Teste pareado AgenticML vs FLAML no mesmo dataset, pareando por semente.
    Roda Wilcoxon (signed-rank) e t pareado. Diferença = AgenticML − FLAML.
    """
    from scipy import stats

    sub = long_df[long_df["dataset"] == dataset]
    piv = sub.pivot_table(index="seed", columns="system", values="metric")
    if not {"AgenticML", "FLAML"}.issubset(piv.columns):
        return {"dataset": dataset, "error": "faltam resultados de AgenticML ou FLAML"}

    piv = piv.dropna(subset=["AgenticML", "FLAML"])
    n = len(piv)
    if n < 2:
        return {"dataset": dataset, "error": f"apenas {n} semente(s) pareada(s)"}

    diff = (piv["AgenticML"] - piv["FLAML"]).to_numpy(dtype=float)
    mean_diff = float(diff.mean())

    try:
        w_stat, w_p = stats.wilcoxon(piv["AgenticML"], piv["FLAML"])
    except ValueError:  # diffs todas zero, ou n pequeno demais
        w_stat, w_p = float("nan"), float("nan")
    t_stat, t_p = stats.ttest_rel(piv["AgenticML"], piv["FLAML"])

    if np.isnan(w_p) or w_p >= alpha:
        verdict = (
            f"Diferença média de {mean_diff:+.4f} NÃO é estatisticamente significativa "
            f"(Wilcoxon p={w_p:.3f}, t pareado p={t_p:.3f}, n={n}). "
            f"A vantagem observada pode decorrer de variabilidade amostral."
        )
    else:
        winner = "AgenticML" if mean_diff > 0 else "FLAML"
        verdict = (
            f"{winner} é significativamente melhor: diferença média {mean_diff:+.4f} "
            f"(Wilcoxon p={w_p:.3f}, t pareado p={t_p:.3f}, n={n})."
        )

    return {
        "dataset": dataset,
        "n_pairs": n,
        "mean_agenticml": round(float(piv["AgenticML"].mean()), 4),
        "mean_flaml": round(float(piv["FLAML"].mean()), 4),
        "mean_diff": round(mean_diff, 4),
        "wilcoxon_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
        "wilcoxon_p": round(float(w_p), 4) if not np.isnan(w_p) else None,
        "ttest_stat": round(float(t_stat), 4),
        "ttest_p": round(float(t_p), 4),
        "significant": bool(not np.isnan(w_p) and w_p < alpha),
        "verdict": verdict,
    }


# ─── Explicabilidade formal: SHAP ──────────────────────────────────────────
# Aprofunda o eixo de XAI: em vez das importâncias nativas do modelo
# (feature_importances_/coef_ que o Agente 6 já reporta), calcula valores SHAP
# sobre o MESMO espaço de features transformado pelo Agente 3.

def _pick_shap_explainer(model, model_name: str, background: pd.DataFrame, task: str):
    """Escolhe o explainer conforme a família do modelo vencedor."""
    import shap

    name = (model_name or "").lower()
    if any(k in name for k in ("forest", "xgboost", "boost", "tree", "gradient")):
        return shap.TreeExplainer(model), "TreeExplainer"
    if any(k in name for k in ("logistic", "linear", "ridge", "lasso")):
        return shap.LinearExplainer(model, background), "LinearExplainer"
    # SVM/SVR e demais: KernelExplainer (model-agnostic, mais lento)
    if task == CLASSIFICATION and hasattr(model, "predict_proba"):
        fn = model.predict_proba
    else:
        fn = model.predict
    return shap.KernelExplainer(fn, background), "KernelExplainer"


def _global_shap_importance(shap_values: Any, feat_cols: list[str]) -> list[dict]:
    """
    Reduz os valores SHAP a uma importância global = média(|SHAP|) por feature.
    Trata os formatos: lista por classe, (n, f), (n, f, classes), (classes, n, f).
    """
    n_feat = len(feat_cols)

    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(np.asarray(a)) for a in shap_values], axis=0)
        imp = stacked.mean(axis=tuple(range(stacked.ndim - 1))) if stacked.shape[-1] == n_feat \
            else stacked.mean(axis=(0, 1))
    else:
        arr = np.abs(np.asarray(shap_values))
        if arr.ndim == 1:
            imp = arr
        else:
            feat_axes = [i for i, s in enumerate(arr.shape) if s == n_feat]
            fax = feat_axes[-1] if feat_axes else 1
            imp = arr.mean(axis=tuple(i for i in range(arr.ndim) if i != fax))

    imp = np.asarray(imp, dtype=float).ravel()[:n_feat]
    order = np.argsort(imp)[::-1]
    return [
        {"feature": feat_cols[i], "mean_abs_shap": round(float(imp[i]), 6)}
        for i in order
    ]


def _rank_agreement(shap_imp: list[dict], native_imp: list[dict]) -> dict:
    """Correlação de Spearman entre o ranking SHAP e o ranking de importância nativa."""
    from scipy import stats

    if not native_imp:
        return {"spearman": None, "note": "modelo sem importância nativa (ex.: SVM rbf)"}

    shap_rank = {d["feature"]: i for i, d in enumerate(shap_imp)}
    nat_rank = {d["feature"]: i for i, d in enumerate(native_imp)}
    common = [f for f in shap_rank if f in nat_rank]
    if len(common) < 3:
        return {"spearman": None, "note": f"apenas {len(common)} feature(s) em comum"}

    rho, p = stats.spearmanr(
        [shap_rank[f] for f in common], [nat_rank[f] for f in common]
    )
    return {
        "spearman": round(float(rho), 4),
        "p_value": round(float(p), 4),
        "n_features": len(common),
        "top5_shap": [d["feature"] for d in shap_imp[:5]],
        "top5_native": [d["feature"] for d in native_imp[:5]],
    }


def shap_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    prompt: str,
    hint: str | None,
    seed: int = 42,
    balance_strategy: str = "auto",
    max_background: int = 100,
    max_explain: int = 200,
) -> dict:
    """
    Roda o pipeline no train, extrai o modelo vencedor e calcula SHAP no test
    (mesmo espaço de features do treino). Compara com a importância nativa que o
    Agente 6 já reporta.

    Retorna dict com:
      model, explainer, shap_importance (lista ordenada), native_importance,
      rank_agreement (Spearman SHAP × nativa), e — para plots —
      shap_values + X_explained + feature_names.
    """
    import shap  # noqa: F401  (garante dependência presente com erro claro)

    state = initial_state(
        dataset_bytes=train_df.to_csv(index=False).encode(),
        filename="train.csv",
        prompt=prompt,
        problem_type_hint=hint,
        random_state=seed,
        balance_strategy=balance_strategy,
    )
    final = build_graph().invoke(state)
    if (final.get("output") or {}).get("status") != "done":
        return {"status": final.get("status"), "error": "pipeline não concluiu"}

    features = final["features"]
    training = final["training"]
    problem_type = (final.get("interpretation") or {}).get("problem_type", CLASSIFICATION)
    task = REGRESSION if problem_type in ("regression", "time_series") else CLASSIFICATION

    model = joblib.load(io.BytesIO(training["model_artifact"]))
    feat_cols: list[str] = features["feature_columns"]
    fitted: dict = features["fitted_params"]

    # Reaplica as transformações do Agente 3 aos dois conjuntos (mesmo caminho do /predict)
    X_train_t = _align_features(_apply_transformations(train_df.copy(), fitted), feat_cols)
    X_test_t = _align_features(_apply_transformations(test_df.copy(), fitted), feat_cols)

    bg = X_train_t.sample(n=min(max_background, len(X_train_t)), random_state=seed)
    X_expl = X_test_t.sample(n=min(max_explain, len(X_test_t)), random_state=seed)

    explainer, explainer_name = _pick_shap_explainer(
        model, training["model_selected"], bg, task
    )
    if explainer_name == "KernelExplainer":
        shap_values = explainer.shap_values(X_expl, nsamples=100, silent=True)
    else:
        shap_values = explainer.shap_values(X_expl)

    shap_imp = _global_shap_importance(shap_values, feat_cols)
    native_imp = (final.get("output") or {}).get("feature_importances", [])
    agreement = _rank_agreement(shap_imp, native_imp)

    return {
        "status": "done",
        "model": training["model_selected"],
        "explainer": explainer_name,
        "problem_type": problem_type,
        "shap_importance": shap_imp,
        "native_importance": native_imp,
        "rank_agreement": agreement,
        "shap_values": shap_values,
        "X_explained": X_expl,
        "feature_names": feat_cols,
    }
