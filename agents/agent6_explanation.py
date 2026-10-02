"""
Agente 6 — Previsão e Explicação
Spec: seções 6.6 e 7.3

Responsabilidade: Gerar previsões e o relatório final explicando todas as
decisões do pipeline em dois níveis (técnico e acessível).

Modos de operação:
  1. Modo normal (caminho feliz): gera previsões + relatório com LLM
  2. Modo erro (curto-circuito, spec 7.3): formata o erro para o usuário

Invariantes (spec 6.6):
  - Relatório sempre com duas versões: technical e accessible
  - Explicação técnica referencia decisões de todos os agentes anteriores
  - Explicação acessível em português claro, sem jargões
  - Previsões entregues em JSON e CSV
  - Edge case 5: prediction_data com schema diferente do treino → curto-circuito
  - Edge case 6: clustering → não gera métricas supervisionadas
"""

import csv
import io
import logging
import joblib
import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.agent3_features import apply_feature_op
from core.llm import get_llm
from core.state import AgenticMLState

AGENT_NAME = "Agente 6 — Previsão e Explicação"
logger = logging.getLogger(__name__)


# ─── Pydantic model para output estruturado do LLM ───────────────────────────

class ExplanationResult(BaseModel):
    technical: str = Field(
        description=(
            "Explicação técnica detalhada para audiência de ML: métricas, transformações, "
            "comparação de candidatos, importância de features. Mínimo 150 palavras."
        )
    )
    accessible: str = Field(
        description=(
            "Explicação em linguagem clara sem jargões para usuários sem background técnico. "
            "Foco no que os resultados significam. Máximo 100 palavras."
        )
    )


# ─── Exceção interna para incompatibilidade de schema ────────────────────────

class _SchemaError(Exception):
    pass


# ─── Entry point do nó LangGraph ─────────────────────────────────────────────

def run_agent6(state: AgenticMLState) -> dict:
    """
    Nó LangGraph do Agente 6.
    Detecta automaticamente modo erro (curto-circuito) ou caminho feliz.
    """
    if state.get("status") == "error":  # type: ignore[call-overload]
        return _handle_error_mode(state)
    return _handle_happy_path(state)


# ─── Modo erro (spec 7.3) ─────────────────────────────────────────────────────

def _handle_error_mode(state: AgenticMLState) -> dict:
    """Formata o payload de erro para o usuário (spec 7.3)."""
    error = state.get("error") or {}  # type: ignore[call-overload]
    failed_at = state.get("failed_at") or "Agente desconhecido"  # type: ignore[call-overload]

    return {
        "output": {
            "status": "error",
            "failed_at": failed_at,
            "explanation": {
                "technical": error.get("technical", "Erro não especificado."),
                "accessible": error.get(
                    "accessible",
                    "Ocorreu um problema durante o processamento. "
                    "Verifique o arquivo enviado e tente novamente.",
                ),
            },
        },
        "status": "done",
    }


# ─── Caminho feliz ────────────────────────────────────────────────────────────

def _handle_happy_path(state: AgenticMLState) -> dict:
    """Gera previsões e relatório final após pipeline completo (spec 6.6)."""
    inspection: dict = state.get("inspection") or {}  # type: ignore[call-overload]
    quality: dict = state.get("quality") or {}  # type: ignore[call-overload]
    features: dict = state.get("features") or {}  # type: ignore[call-overload]
    interpretation: dict = state.get("interpretation") or {}  # type: ignore[call-overload]
    training: dict = state.get("training") or {}  # type: ignore[call-overload]

    problem_type: str = interpretation.get("problem_type", "classification")

    # Extrai importâncias de features do modelo treinado
    importances = _get_feature_importances(training, features, problem_type)

    # Gera previsões se prediction_data foi fornecido
    predictions: dict | None = None
    prediction_data_bytes = state.get("prediction_data")  # type: ignore[call-overload]
    filename: str = state.get("filename", "dataset.csv")  # type: ignore[call-overload]

    if prediction_data_bytes:
        try:
            predictions = _generate_predictions(
                prediction_data_bytes, filename, features, training
            )
            logger.info("[Agente 6] Previsões geradas: %d registros.", predictions.get("count", 0))
        except _SchemaError as exc:
            # Edge case 5: incompatibilidade de schema → curto-circuito
            return _error(str(exc))
        except Exception as exc:
            logger.warning("[Agente 6] Falha ao gerar previsões: %s", exc)
            predictions = {"error": str(exc)}

    # Gera explicação via LLM (com fallback determinístico)
    try:
        explanation = _llm_explain(
            inspection, quality, features, interpretation, training, importances, predictions
        )
    except Exception as exc:
        logger.warning("[Agente 6] LLM indisponível, usando fallback: %s", exc)
        explanation = _fallback_explain(
            inspection, quality, features, interpretation, training, importances, predictions
        )

    logger.info("[Agente 6] Pipeline concluído. Modelo: %s", training.get("model_selected"))

    return {
        "output": {
            "status": "done",
            "predictions": predictions,
            "feature_importances": importances,
            "explanation": {
                "technical": explanation.technical,
                "accessible": explanation.accessible,
            },
        },
        "status": "done",
    }


# ─── Importância de features ──────────────────────────────────────────────────

def _get_feature_importances(
    training: dict, features: dict, problem_type: str
) -> list[dict]:
    """
    Extrai as top-10 features por importância do modelo treinado.
    Suporta tree-based (feature_importances_) e lineares (coef_).
    Edge case 6: clustering/anomaly_detection → retorna lista vazia.
    """
    if problem_type in ("clustering", "anomaly_detection"):
        return []

    model_bytes = training.get("model_artifact")
    if not model_bytes:
        return []

    feature_columns: list[str] = features.get("feature_columns", [])
    if not feature_columns:
        return []

    try:
        model = joblib.load(io.BytesIO(model_bytes))
    except Exception:
        return []

    importances_arr: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances_arr = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        importances_arr = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances_arr is None:
        return []

    n = min(len(importances_arr), len(feature_columns))
    pairs = sorted(
        zip(feature_columns[:n], importances_arr[:n].tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"feature": f, "importance": round(v, 6)} for f, v in pairs[:10]]


# ─── Geração de previsões ─────────────────────────────────────────────────────

def _generate_predictions(
    data_bytes: bytes,
    original_filename: str,
    features: dict,
    training: dict,
) -> dict:
    """
    Aplica as transformações do treino a novos dados e gera previsões.
    Edge case 5: schema mismatch → _SchemaError.
    """
    df_pred = _parse_bytes(data_bytes, original_filename)
    fitted_params: dict = features.get("fitted_params", {})
    original_feature_cols: list[str] = fitted_params.get("original_feature_columns", [])
    target_col: str | None = fitted_params.get("target_column")

    # Validação de schema (edge case 5)
    if original_feature_cols:
        pred_col_set = set(df_pred.columns)
        required = set(original_feature_cols)
        missing = required - pred_col_set
        if missing:
            raise _SchemaError(
                f"Incompatibilidade de schema nos dados de previsão: colunas ausentes: "
                f"{sorted(missing)}. "
                f"Os dados devem conter as mesmas colunas do dataset de treino "
                f"(sem a coluna target '{target_col}')."
            )

    df_transformed = _apply_transformations(df_pred, fitted_params)
    feature_columns: list[str] = features.get("feature_columns", [])
    X = _align_features(df_transformed, feature_columns)

    model_bytes = training.get("model_artifact")
    if not model_bytes:
        raise ValueError("Artefato do modelo não encontrado no State.")

    model = joblib.load(io.BytesIO(model_bytes))
    raw_preds = model.predict(X.values)

    # Decodifica labels se o target foi codificado por LabelEncoder
    label_enc = training.get("label_encoder_target")
    if label_enc is not None and hasattr(label_enc, "inverse_transform"):
        try:
            values: list = label_enc.inverse_transform(raw_preds.astype(int)).tolist()
        except Exception:
            values = raw_preds.tolist()
    else:
        values = raw_preds.tolist()

    # Serializa CSV
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["prediction"])
    for v in values:
        writer.writerow([v])

    return {
        "values": values,
        "count": len(values),
        "format": "json+csv",
        "csv": csv_buf.getvalue(),
    }


def _parse_bytes(data_bytes: bytes, filename: str) -> pd.DataFrame:
    """Lê bytes de arquivo (CSV/XLSX/XLS) em um DataFrame."""
    import chardet
    import csv as csv_mod

    fname = filename.lower()
    if fname.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data_bytes), engine="openpyxl")
    if fname.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data_bytes), engine="xlrd")

    detected = chardet.detect(data_bytes[:10_000])
    encoding = detected.get("encoding") or "utf-8"
    text = data_bytes.decode(encoding, errors="replace")
    try:
        dialect = csv_mod.Sniffer().sniff(text[:4096])
        sep = dialect.delimiter
    except csv_mod.Error:
        sep = ","
    return pd.read_csv(io.StringIO(text), sep=sep)


def _apply_transformations(df: pd.DataFrame, fitted_params: dict) -> pd.DataFrame:
    """Re-aplica as transformações fittadas do treino a novos dados."""
    df = df.copy()
    target_col = fitted_params.get("target_column")

    # Remove target e colunas descartadas
    cols_to_remove = list(fitted_params.get("columns_to_drop", []))
    if target_col and target_col in df.columns:
        cols_to_remove.append(target_col)
    existing_drops = [c for c in cols_to_remove if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    # Imputação
    for col, imputer in fitted_params.get("imputers", {}).items():
        if col in df.columns:
            df[[col]] = imputer.transform(df[[col]])

    # Criação de features — mesma receita e mesma ordem do Agente 3 (depois da imputação,
    # antes do encoding/scaling), via apply_feature_op compartilhado com agent3_features.
    for spec in fitted_params.get("feature_creation", []):
        c1, c2 = spec["column_1"], spec["column_2"]
        if c1 in df.columns and c2 in df.columns:
            df[spec["new_column"]] = apply_feature_op(df[c1], df[c2], spec["operation"])

    # Label Encoding de features (alta cardinalidade)
    for col, le in fitted_params.get("label_encoders", {}).items():
        if col in df.columns:
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda v, k=known, encoder=le: encoder.transform([v])[0] if v in k else -1
            )

    # One-Hot Encoding — reconstrói dummies e reindexo para as colunas do treino
    ohe_produced = fitted_params.get("ohe_produced_columns", {})
    for col, dummy_cols in ohe_produced.items():
        if col in df.columns:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True).astype(int)
            dummies = dummies.reindex(columns=dummy_cols, fill_value=0)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Scaling — agrupa colunas pelo mesmo objeto scaler (fittado em conjunto)
    scaler_groups: dict[int, tuple] = {}
    for col, scaler in fitted_params.get("scalers", {}).items():
        key = id(scaler)
        if key not in scaler_groups:
            scaler_groups[key] = (scaler, [])
        scaler_groups[key][1].append(col)

    for scaler_obj, cols in scaler_groups.values():
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            df[valid_cols] = scaler_obj.transform(df[valid_cols])

    return df


def _align_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Reordena e preenche colunas ausentes para corresponder ao conjunto de treino."""
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns].fillna(0)


# ─── Explicação via LLM ───────────────────────────────────────────────────────

_SYSTEM_PROMPT_EXPLAIN = """\
Você é um especialista em Machine Learning gerando o relatório explicativo final do pipeline AgenticML.

Gere DUAS versões de explicação:
1. technical: para audiência técnica (cientistas de dados, engenheiros de ML).
   - Detalhe técnico: métricas, transformações, comparação de modelos, importâncias de features
   - Referencie decisões de TODOS os agentes (Agentes 1 a 5)
   - Mínimo 150 palavras

2. accessible: para usuários sem background técnico.
   - Linguagem clara e direta, sem siglas não explicadas
   - Foco no que os resultados significam para o negócio
   - Máximo 100 palavras

REGRAS:
- Não invente dados — use apenas o que está no contexto fornecido
- Escreva em português do Brasil
- Se não houver previsões, não mencione previsões
"""


def _build_explain_message(
    inspection: dict,
    quality: dict,
    features: dict,
    interpretation: dict,
    training: dict,
    importances: list[dict],
    predictions: dict | None,
) -> str:
    schema = inspection.get("schema", {})
    shape = schema.get("shape", [0, 0])

    issues = quality.get("issues", [])
    issues_str = (
        "; ".join(f"{i['type']} (severidade={i['severity']})" for i in issues) or "nenhum"
    )

    transforms = features.get("transformations_applied", [])
    trans_str = "; ".join(f"{t['type']} em {t['columns']}" for t in transforms) or "nenhuma"

    problem_type = interpretation.get("problem_type", "?")
    ptype_just = interpretation.get("justification", "")
    diverged = interpretation.get("diverged_from_user", False)

    model = training.get("model_selected", "?")
    metric = training.get("metric_used", "?")
    metric_val = training.get("metric_value", "?")
    candidates = training.get("candidate_results", [])
    cands_str = "; ".join(f"{c['model']}={c['score']:.4f}" for c in candidates)
    training_just = training.get("justification", "")

    balancing = training.get("balancing") or {}
    balancing_str = (
        f"{balancing.get('method')} — {balancing.get('reason')}"
        if balancing.get("applied")
        else "nenhum (classes suficientemente equilibradas ou desativado)"
    )

    feats_str = (
        "; ".join(f"{f['feature']}={f['importance']:.4f}" for f in importances[:5])
        or "não disponível (SVM sem kernel linear)"
    )

    rmse_line = f" | RMSE={training['rmse']:.4f}" if "rmse" in training else ""

    if predictions and "count" in predictions:
        preds_info = f"Previsões geradas: {predictions['count']} registros."
    else:
        preds_info = "Nenhum dado de previsão fornecido nesta requisição."

    return (
        f"=== PIPELINE AgenticML ===\n"
        f"Dataset: {shape[0]:,} linhas × {shape[1]} colunas | "
        f"Encoding: {inspection.get('encoding_detected', '?')} | "
        f"Separador: {inspection.get('separator_detected', '?')}\n\n"
        f"[Agente 2 — Qualidade] Issues: {issues_str}\n"
        f"Overall severity: {quality.get('overall_severity', '?')}\n\n"
        f"[Agente 3 — Feature Engineering] Transformações: {trans_str}\n"
        f"Coluna target: {features.get('target_column', '?')}\n\n"
        f"[Agente 4 — Interpretação] Tipo de problema: {problem_type}\n"
        f"Justificativa: {ptype_just}\n"
        f"Divergiu da sugestão do usuário: {diverged}\n\n"
        f"[Agente 5 — Treinamento] Modelo selecionado: {model}\n"
        f"Métrica de seleção: {metric}={metric_val}{rmse_line}\n"
        f"Todos os candidatos avaliados: {cands_str}\n"
        f"Tratamento de desbalanceamento de classes: {balancing_str}\n"
        f"Justificativa de seleção: {training_just}\n\n"
        f"Top features por importância: {feats_str}\n\n"
        f"Previsões: {preds_info}\n\n"
        "Gere o relatório explicativo (technical + accessible)."
    )


def _llm_explain(
    inspection: dict,
    quality: dict,
    features: dict,
    interpretation: dict,
    training: dict,
    importances: list[dict],
    predictions: dict | None,
) -> ExplanationResult:
    llm = get_llm()
    structured = llm.with_structured_output(ExplanationResult)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT_EXPLAIN),
        HumanMessage(
            content=_build_explain_message(
                inspection, quality, features, interpretation, training, importances, predictions
            )
        ),
    ]
    return structured.invoke(messages)


# ─── Fallback determinístico sem LLM ─────────────────────────────────────────

def _fallback_explain(
    inspection: dict,
    quality: dict,
    features: dict,
    interpretation: dict,
    training: dict,
    importances: list[dict],
    predictions: dict | None,
) -> ExplanationResult:
    """Gera explicação template-based quando o LLM está indisponível."""
    schema = inspection.get("schema", {})
    shape = schema.get("shape", [0, 0])
    problem_type = interpretation.get("problem_type", "não identificado")
    model_selected = training.get("model_selected", "?")
    metric_used = training.get("metric_used", "?")
    metric_value = training.get("metric_value", "?")
    n_issues = len(quality.get("issues", []))
    n_transforms = len(features.get("transformations_applied", []))
    candidates = training.get("candidate_results", [])
    cands_str = ", ".join(f"{c['model']} ({c['score']:.4f})" for c in candidates)

    top_features = importances[:3]
    feats_str = ", ".join(f["feature"] for f in top_features) if top_features else "não disponível"

    rmse_part = f" RMSE={training['rmse']:.4f}." if "rmse" in training else ""

    balancing = training.get("balancing") or {}
    balancing_part = (
        f" Desbalanceamento de classes tratado via {balancing.get('method')} "
        f"({balancing.get('reason')})."
        if balancing.get("applied")
        else ""
    )

    technical = (
        f"Pipeline AgenticML concluído. "
        f"Dataset: {shape[0]:,} linhas × {shape[1]} colunas. "
        f"Tipo de problema identificado: {problem_type}. "
        f"Diagnóstico de qualidade (Agente 2): {n_issues} issue(s). "
        f"Feature Engineering (Agente 3): {n_transforms} transformação(ões) aplicada(s). "
        f"Modelo selecionado (Agente 5): {model_selected} com {metric_used}={metric_value}.{rmse_part}{balancing_part} "
        f"Candidatos avaliados: {cands_str}. "
        f"Top features: {feats_str}."
    )

    if predictions and predictions.get("count", 0) > 0:
        technical += f" Previsões: {predictions['count']} registros gerados."

    _metric_labels = {
        "f1_weighted": "precisão de classificação (F1)",
        "r2": "coeficiente de determinação (R²)",
        "silhouette": "coesão dos grupos (Silhouette)",
        "decision_score": "pontuação de anomalia",
    }
    metric_label = _metric_labels.get(metric_used, metric_used)

    accessible = (
        f"O sistema analisou seu dataset com {shape[0]:,} linhas e identificou que se trata de "
        f"um problema de {problem_type}. "
        f"Após organizar os dados ({n_transforms} transformação(ões) aplicada(s)), "
        f"{len(candidates)} modelos foram testados e o {model_selected} foi escolhido "
        f"por apresentar o melhor {metric_label} ({metric_value})."
    )
    if feats_str != "não disponível":
        accessible += f" As variáveis mais importantes são: {feats_str}."
    if predictions and predictions.get("count", 0) > 0:
        accessible += f" Foram geradas {predictions['count']} previsão(ões)."

    return ExplanationResult(technical=technical, accessible=accessible)


# ─── Error helper ─────────────────────────────────────────────────────────────

def _error(message: str) -> dict:
    return {
        "status": "error",
        "failed_at": AGENT_NAME,
        "error": {
            "technical": message,
            "accessible": (
                "O sistema não conseguiu completar a geração de previsões. "
                f"Detalhe: {message}"
            ),
        },
    }
