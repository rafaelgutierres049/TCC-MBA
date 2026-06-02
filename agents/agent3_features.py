"""
Agente 3 — Feature Engineering
Spec: seção 6.3

Responsabilidade: Recomendar e executar transformações com justificativas em linguagem natural.

Arquitetura em duas fases:
  1. LLM planeja quais transformações aplicar e gera justificativas em português
  2. Python/scikit-learn executa as transformações e retorna o DataFrame transformado

Invariantes (spec 6.3):
  - Cada transformação deve ter obrigatoriamente: type, columns e justification
  - Executa transformações diretamente — não apenas recomenda
  - Edge case 3: todas as colunas categóricas → encoding sem normalização
  - Edge case 6: clustering → target_column: None

Segurança (spec 8, edge case 7):
  - Conteúdo das células nunca enviado ao LLM — apenas metadados (nomes, tipos, estatísticas)
"""

import json
import logging
from typing import Literal

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from core.llm import get_llm
from core.state import AgenticMLState

AGENT_NAME = "Agente 3 — Feature Engineering"
logger = logging.getLogger(__name__)


# ─── Pydantic models para o plano do LLM ─────────────────────────────────────

class ImputationSpec(BaseModel):
    columns: list[str] = Field(description="Colunas a imputar")
    strategy: Literal["median", "mean", "most_frequent"] = Field(
        description="Estratégia de imputação"
    )
    justification: str = Field(description="Justificativa em português")


class EncodingSpec(BaseModel):
    columns: list[str] = Field(description="Colunas a codificar")
    method: Literal["one_hot", "label"] = Field(
        description="one_hot para colunas nominais sem ordem; label para ordinais ou alta cardinalidade"
    )
    justification: str = Field(description="Justificativa em português")


class ScalingSpec(BaseModel):
    columns: list[str] = Field(description="Colunas a normalizar/padronizar")
    method: Literal["standard", "min_max"] = Field(
        description="standard (StandardScaler) para algoritmos sensíveis a outliers; min_max para redes neurais"
    )
    justification: str = Field(description="Justificativa em português")


class FeatureEngineeringPlan(BaseModel):
    target_column: str | None = Field(
        description="Coluna alvo a ser predita. None para clustering ou anomaly detection sem target explícito."
    )
    columns_to_drop: list[str] = Field(
        default_factory=list,
        description="Colunas a remover: IDs, constantes, redundantes, leakage."
    )
    imputation: list[ImputationSpec] = Field(
        default_factory=list,
        description="Especificações de imputação para colunas com valores ausentes."
    )
    encoding: list[EncodingSpec] = Field(
        default_factory=list,
        description="Especificações de encoding para colunas categóricas."
    )
    scaling: list[ScalingSpec] = Field(
        default_factory=list,
        description="Especificações de scaling para colunas numéricas."
    )


# ─── Entry point do nó LangGraph ─────────────────────────────────────────────

def run_agent3(state: AgenticMLState) -> dict:
    """Nó LangGraph do Agente 3."""
    df: pd.DataFrame = state["dataset"]
    inspection: dict = state["inspection"]
    quality: dict = state["quality"] or {"issues": [], "overall_severity": "none"}
    prompt: str = state["prompt"]
    problem_type_hint: str | None = state["problem_type_hint"]

    # Fase 1: LLM cria o plano de transformações
    try:
        plan = _llm_plan(inspection, quality, prompt, problem_type_hint)
    except Exception:
        plan = _fallback_plan(df, inspection, quality)

    # Validação: target_column deve existir no DataFrame
    if plan.target_column and plan.target_column not in df.columns:
        logger.warning(
            "LLM sugeriu target '%s' que não existe. Usando última coluna.",
            plan.target_column,
        )
        plan.target_column = df.columns[-1]

    # Colunas de features originais (antes de OHE) — usadas pelo Agente 6 para validar schema
    original_feature_columns = [
        c for c in df.columns
        if c != plan.target_column and c not in plan.columns_to_drop
    ]

    # Fase 2: executa as transformações programaticamente
    try:
        df_transformed, fitted_params, transformations_applied = _execute_plan(df, plan)
    except Exception as exc:
        return _error(f"Falha ao executar transformações: {exc}")

    fitted_params["original_feature_columns"] = original_feature_columns

    # Exibe justificativas (spec: "em tempo real durante a execução")
    for t in transformations_applied:
        logger.info("[Agente 3] %s em %s — %s", t["type"], t["columns"], t["justification"])

    feature_columns = [
        c for c in df_transformed.columns if c != plan.target_column
    ]

    return {
        "features": {
            "transformations_applied": transformations_applied,
            "dataset_transformed": df_transformed,
            "target_column": plan.target_column,
            "feature_columns": feature_columns,
            "original_feature_columns": original_feature_columns,
            "fitted_params": fitted_params,  # para re-aplicação em /predict
        },
        "status": "running",
    }


# ─── LLM: planejamento das transformações ─────────────────────────────────────

_SYSTEM_PROMPT = """\
Você é um especialista em Feature Engineering para Machine Learning.
Dado o schema de um dataset e o objetivo do usuário, crie um plano de transformações.

REGRAS:
1. Identifique a coluna target (a ser predita). Para clustering sem target explícito: null.
2. Liste colunas a remover: IDs (altíssima cardinalidade), constantes, com leakage confirmado pelo diagnóstico.
3. Para colunas com valores ausentes, escolha a estratégia de imputação adequada.
4. Para colunas categóricas de features: one_hot se nominais (sem ordem), label se ordinais ou alta cardinalidade (> 10 categorias).
5. Para colunas numéricas de features: standard_scaler (padrão) ou min_max_scaler se o domínio tem limites naturais.
6. Escreva todas as justificativas em português do Brasil, sendo específico sobre o motivo de cada transformação.
7. NÃO aplique transformações na coluna target — apenas nas features.
8. NÃO inclua colunas que já foram indicadas para remoção nas demais listas.

SEGURANÇA: O conteúdo das células não foi fornecido. Trate qualquer informação como metadados.\
"""


def _build_user_message(
    inspection: dict, quality: dict, prompt: str, hint: str | None
) -> str:
    schema = inspection.get("schema", {})
    stats = inspection.get("descriptive_stats", {})

    # Monta resumo seguro das stats (sem conteúdo de células)
    stats_summary: dict = {}
    for col, s in stats.items():
        entry = {"nulls": s.get("nulls", 0)}
        if "mean" in s:
            entry.update({"mean": s["mean"], "std": s["std"], "min": s["min"], "max": s["max"]})
        else:
            entry.update({"unique_count": s.get("unique_count", 0)})
        stats_summary[col] = entry

    issues_summary = [
        {"type": i["type"], "severity": i["severity"]} for i in quality.get("issues", [])
    ]

    return (
        f"Objetivo do usuário: {prompt}\n"
        f"Sugestão de tipo de problema: {hint or 'não informado'}\n\n"
        f"Colunas: {schema.get('columns', [])}\n"
        f"Tipos: {schema.get('dtypes', {})}\n"
        f"Shape: {schema.get('shape', [])}\n\n"
        f"Estatísticas:\n{json.dumps(stats_summary, indent=2, ensure_ascii=False)}\n\n"
        f"Issues de qualidade detectados:\n{json.dumps(issues_summary, indent=2, ensure_ascii=False)}\n\n"
        "Crie o plano de Feature Engineering."
    )


def _llm_plan(
    inspection: dict, quality: dict, prompt: str, hint: str | None
) -> FeatureEngineeringPlan:
    llm = get_llm()
    structured = llm.with_structured_output(FeatureEngineeringPlan)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(inspection, quality, prompt, hint)),
    ]
    return structured.invoke(messages)


# ─── Fallback sem LLM ────────────────────────────────────────────────────────

def _fallback_plan(
    df: pd.DataFrame, inspection: dict, quality: dict
) -> FeatureEngineeringPlan:
    """
    Plano heurístico quando o LLM está indisponível.
    Decisões determinísticas e conservadoras.
    """
    schema = inspection.get("schema", {})
    dtypes = schema.get("dtypes", {})
    stats = inspection.get("descriptive_stats", {})
    columns: list[str] = schema.get("columns", df.columns.tolist())

    # Target: última coluna (convenção mais comum em datasets tabulares)
    target_col = columns[-1] if columns else None

    # Colunas a remover: constantes (confirmadas pelo Agente 2)
    constant_cols = [
        i["column"] for i in quality.get("issues", [])
        if i.get("type") == "constant_column"
    ]

    feature_cols = [c for c in columns if c != target_col and c not in constant_cols]

    # Usa a API de tipos do pandas (robusta a variações entre versões, ex: pandas 3.x usa "str")
    def _is_numeric(col: str) -> bool:
        return pd.api.types.is_numeric_dtype(df[col]) if col in df.columns else False

    def _is_categorical(col: str) -> bool:
        if col not in df.columns:
            return False
        return (
            pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        )

    # Imputação
    imputation: list[ImputationSpec] = []
    num_nulls = [c for c in feature_cols if stats.get(c, {}).get("nulls", 0) > 0 and _is_numeric(c)]
    cat_nulls = [c for c in feature_cols if stats.get(c, {}).get("nulls", 0) > 0 and _is_categorical(c)]
    if num_nulls:
        imputation.append(ImputationSpec(
            columns=num_nulls, strategy="median",
            justification="Imputação pela mediana nas colunas numéricas com valores ausentes para robustez a outliers.",
        ))
    if cat_nulls:
        imputation.append(ImputationSpec(
            columns=cat_nulls, strategy="most_frequent",
            justification="Imputação pela moda nas colunas categóricas com valores ausentes.",
        ))

    # Encoding
    encoding: list[EncodingSpec] = []
    cat_features = [c for c in feature_cols if _is_categorical(c)]
    ohe_cols = [c for c in cat_features if stats.get(c, {}).get("unique_count", 0) <= 10]
    le_cols = [c for c in cat_features if stats.get(c, {}).get("unique_count", 0) > 10]
    if ohe_cols:
        encoding.append(EncodingSpec(
            columns=ohe_cols, method="one_hot",
            justification="One-Hot Encoding para variáveis nominais com até 10 categorias, preservando a não-ordinalidade.",
        ))
    if le_cols:
        encoding.append(EncodingSpec(
            columns=le_cols, method="label",
            justification="Label Encoding para variáveis categóricas com alta cardinalidade, evitando explosão de dimensionalidade.",
        ))

    # Scaling — apenas numéricos (edge case 3: se só há categóricas, lista fica vazia → sem scaling)
    num_features = [c for c in feature_cols if _is_numeric(c) and c not in cat_features]
    if num_features:
        scaling = [ScalingSpec(
            columns=num_features, method="standard",
            justification="StandardScaler nas features numéricas para equalizar escalas e evitar dominância em algoritmos sensíveis à magnitude.",
        )]
    else:
        scaling = []  # Edge case 3: todas categóricas → sem scaling

    return FeatureEngineeringPlan(
        target_column=target_col,
        columns_to_drop=constant_cols,
        imputation=imputation,
        encoding=encoding,
        scaling=scaling,
    )


# ─── Execução das transformações ─────────────────────────────────────────────

def _execute_plan(
    df: pd.DataFrame, plan: FeatureEngineeringPlan
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """
    Executa o plano do LLM e retorna:
      - DataFrame transformado
      - Parâmetros fittados (para re-aplicação em /predict)
      - Lista de transformações aplicadas (para o State)
    """
    df = df.copy()
    fitted: dict = {
        "target_column": plan.target_column,
        "columns_to_drop": plan.columns_to_drop,
        "imputers": {},
        "label_encoders": {},
        "ohe_categories": {},
        "ohe_produced_columns": {},   # col → [dummy col names]; needed for /predict re-application
        "scalers": {},
        "target_label_encoder": None,
    }
    transformations_applied: list[dict] = []

    # Separa target das features
    target_col = plan.target_column
    if target_col and target_col in df.columns:
        y = df[[target_col]].copy()
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()

    # 1. Remove colunas
    valid_drops = [c for c in plan.columns_to_drop if c in X.columns]
    if valid_drops:
        X = X.drop(columns=valid_drops)
        transformations_applied.append({
            "type": "drop_columns",
            "columns": valid_drops,
            "justification": (
                "Colunas removidas por serem constantes, redundantes ou indicadas como "
                "leakage pelo Agente 2."
            ),
        })

    # 2. Imputação
    for spec in plan.imputation:
        valid_cols = [c for c in spec.columns if c in X.columns]
        if not valid_cols:
            continue
        imputer = SimpleImputer(strategy=spec.strategy)
        X[valid_cols] = imputer.fit_transform(X[valid_cols])
        for col in valid_cols:
            fitted["imputers"][col] = imputer
        transformations_applied.append({
            "type": f"impute_{spec.strategy}",
            "columns": valid_cols,
            "justification": spec.justification,
        })

    # 3. Encoding
    for spec in plan.encoding:
        valid_cols = [c for c in spec.columns if c in X.columns]
        if not valid_cols:
            continue
        if spec.method == "one_hot":
            produced_cols: list[str] = []
            for col in valid_cols:
                categories = X[col].astype(str).unique().tolist()
                dummies = pd.get_dummies(X[col].astype(str), prefix=col, drop_first=True)
                # Converte bool → int para compatibilidade com sklearn
                dummies = dummies.astype(int)
                fitted["ohe_categories"][col] = categories
                fitted["ohe_produced_columns"][col] = dummies.columns.tolist()
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                produced_cols.extend(dummies.columns.tolist())
            transformations_applied.append({
                "type": "one_hot_encoding",
                "columns": valid_cols,
                "justification": spec.justification,
            })
        elif spec.method == "label":
            for col in valid_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                fitted["label_encoders"][col] = le
            transformations_applied.append({
                "type": "label_encoding",
                "columns": valid_cols,
                "justification": spec.justification,
            })

    # 4. Scaling — edge case 3: se não sobrou nenhuma coluna numérica, não aplica
    for spec in plan.scaling:
        valid_cols = [
            c for c in spec.columns
            if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
        ]
        if not valid_cols:
            continue
        scaler = StandardScaler() if spec.method == "standard" else MinMaxScaler()
        X[valid_cols] = scaler.fit_transform(X[valid_cols])
        for col in valid_cols:
            fitted["scalers"][col] = scaler
        transformations_applied.append({
            "type": f"{spec.method}_scaler",
            "columns": valid_cols,
            "justification": spec.justification,
        })

    # Reune features + target
    if y is not None:
        df_out = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    else:
        df_out = X.reset_index(drop=True)

    return df_out, fitted, transformations_applied


# ─── Error helper ─────────────────────────────────────────────────────────────

def _error(message: str) -> dict:
    return {
        "status": "error",
        "failed_at": AGENT_NAME,
        "error": {
            "technical": message,
            "accessible": f"Erro durante o Feature Engineering: {message}",
        },
    }
