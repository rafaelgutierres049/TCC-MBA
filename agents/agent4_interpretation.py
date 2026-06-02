"""
Agente 4 — Interpretação de Linguagem Natural
Spec: seção 6.4

Responsabilidade: Interpretar o prompt + dados inspecionados para inferir o tipo de problema.

Invariantes (spec 6.4):
  - Se divergir da sugestão do usuário → diverged_from_user: true + continua (NÃO bloqueia)
  - Se não identificar com confiança → curto-circuito solicitando mais informações
  - justification é obrigatório em qualquer cenário

Edge cases (spec 8):
  - #2: prompt vazio ou muito vago → curto-circuito

Tipos suportados: classification | regression | clustering | time_series | anomaly_detection
"""

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.llm import get_llm
from core.state import AgenticMLState

AGENT_NAME = "Agente 4 — Interpretação de Linguagem Natural"

SUPPORTED_TYPES = frozenset(
    {"classification", "regression", "clustering", "time_series", "anomaly_detection"}
)

# Palavras-chave para fallback heurístico
_KEYWORDS: dict[str, list[str]] = {
    "classification": ["classif", "prever", "prediz", "prevê", "cancelar", "churn",
                       "fraude", "diagnos", "spam", "aprovação", "detecção"],
    "regression":    ["estimar", "estima", "regressa", "valor", "preço", "quantidade",
                      "quanto", "quanto custa", "prever numericamente"],
    "clustering":    ["agrupar", "segmentar", "cluster", "grupo", "perfil", "similar"],
    "time_series":   ["série temporal", "séries temporais", "previsão de demanda",
                      "tendência", "sazonalidade", "forecast", "próximos meses"],
    "anomaly_detection": ["anomalia", "outlier", "fraude", "irregularidade", "detecção de"],
}


# ─── Pydantic model para output estruturado do LLM ───────────────────────────

class InterpretationResult(BaseModel):
    problem_type: Literal[
        "classification", "regression", "clustering", "time_series", "anomaly_detection"
    ] = Field(description="Tipo de problema de ML identificado")
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high: inequivocamente identificado; "
            "medium: identificado com razoável certeza; "
            "low: ambíguo ou insuficientemente especificado"
        )
    )
    justification: str = Field(
        description=(
            "Justificativa em português do Brasil explicando como o prompt e as "
            "características do dataset levaram à identificação do tipo de problema. "
            "Mencione a coluna target identificada e por que ela caracteriza este tipo."
        )
    )
    diverged_from_user: bool = Field(
        description="True se o tipo inferido difere da sugestão fornecida pelo usuário."
    )


# ─── Entry point do nó LangGraph ─────────────────────────────────────────────

def run_agent4(state: AgenticMLState) -> dict:
    """Nó LangGraph do Agente 4."""
    prompt: str = state["prompt"]

    # Edge case 2: prompt vazio ou muito vago (spec 8)
    if not prompt or len(prompt.strip()) < 10:
        return _error(
            "O prompt está vazio ou muito vago. Por favor, descreva com mais detalhes "
            "o que você deseja analisar (ex: 'Prever se um cliente vai cancelar o contrato')."
        )

    hint: str | None = state.get("problem_type_hint")  # type: ignore[call-overload]
    inspection: dict = state["inspection"] or {}
    quality: dict = state["quality"] or {}
    features: dict = state["features"] or {}

    # Chama o LLM (com fallback heurístico se indisponível)
    try:
        result = _llm_interpret(prompt, hint, inspection, quality, features)
    except Exception:
        result = _fallback_interpret(prompt, hint)

    # Se confiança baixa → curto-circuito (spec 6.4)
    if result.confidence == "low":
        return _error(
            f"Não foi possível identificar o tipo de problema com confiança suficiente. "
            f"Justificativa: {result.justification} "
            f"Por favor, refine o prompt especificando o objetivo e a variável a ser analisada."
        )

    return {
        "interpretation": result.model_dump(),
        "status": "running",
    }


# ─── LLM: interpretação do problema ──────────────────────────────────────────

_SYSTEM_PROMPT = """\
Você é um especialista em Machine Learning que interpreta objetivos de usuários e características
de datasets para identificar o tipo de problema analítico adequado.

Tipos disponíveis: classification | regression | clustering | time_series | anomaly_detection

DIRETRIZES:
- classification: prever uma categoria/classe (binária ou multi-classe)
- regression: prever um valor numérico contínuo
- clustering: agrupar registros por similaridade (sem variável target explícita)
- time_series: prever valores futuros baseados em sequência temporal
- anomaly_detection: identificar registros atípicos ou fraudulentos

REGRAS:
1. Analise o prompt, a coluna target identificada pelo Agente 3 e as características do dataset.
2. Se o prompt solicita classificar/prever categoria → classification.
3. Se o prompt solicita estimar valor numérico → regression.
4. Use confidence="low" apenas quando o problema for genuinamente ambíguo após análise cuidadosa.
5. Escreva justification em português do Brasil, mencionando evidências concretas.
6. Se o tipo inferido difere da sugestão do usuário, indique diverged_from_user=true e explique.\
"""


def _build_user_message(
    prompt: str,
    hint: str | None,
    inspection: dict,
    quality: dict,
    features: dict,
) -> str:
    schema = inspection.get("schema", {})
    target = features.get("target_column", "não identificada")
    feat_cols = features.get("feature_columns", [])
    issues = [{"type": i["type"], "severity": i["severity"]}
              for i in quality.get("issues", [])]

    return (
        f"Objetivo do usuário: {prompt}\n"
        f"Sugestão de tipo (usuário): {hint or 'não informado'}\n\n"
        f"Coluna target identificada pelo Agente 3: {target}\n"
        f"Colunas de features: {feat_cols}\n"
        f"Schema: colunas={schema.get('columns', [])}, tipos={schema.get('dtypes', {})}\n"
        f"Shape: {schema.get('shape', [])}\n\n"
        f"Issues de qualidade: {json.dumps(issues, ensure_ascii=False)}\n\n"
        "Identifique o tipo de problema."
    )


def _llm_interpret(
    prompt: str,
    hint: str | None,
    inspection: dict,
    quality: dict,
    features: dict,
) -> InterpretationResult:
    llm = get_llm()
    structured = llm.with_structured_output(InterpretationResult)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(prompt, hint, inspection, quality, features)),
    ]
    return structured.invoke(messages)


# ─── Fallback heurístico sem LLM ─────────────────────────────────────────────

def _fallback_interpret(prompt: str, hint: str | None) -> InterpretationResult:
    """
    Detecta o tipo de problema por palavras-chave quando o LLM está indisponível.
    Prioridade: hint do usuário → keywords no prompt → classification (default conservador).
    """
    p = prompt.lower()

    # Se hint é válido, respeita
    if hint in SUPPORTED_TYPES:
        return InterpretationResult(
            problem_type=hint,  # type: ignore[arg-type]
            confidence="medium",
            justification=(
                f"Tipo '{hint}' adotado conforme sugestão do usuário (modo fallback sem LLM). "
                "Recomenda-se revisar com LLM ativo para maior precisão."
            ),
            diverged_from_user=False,
        )

    # Busca por keywords
    for ptype, keywords in _KEYWORDS.items():
        if any(kw in p for kw in keywords):
            return InterpretationResult(
                problem_type=ptype,  # type: ignore[arg-type]
                confidence="medium",
                justification=(
                    f"Tipo '{ptype}' identificado por palavras-chave no prompt (modo fallback sem LLM)."
                ),
                diverged_from_user=(hint is not None and hint != ptype),
            )

    # Default conservador
    return InterpretationResult(
        problem_type="classification",
        confidence="medium",
        justification=(
            "Nenhuma palavra-chave específica identificada no prompt. "
            "Tipo 'classification' adotado como padrão conservador (modo fallback sem LLM)."
        ),
        diverged_from_user=(hint is not None and hint != "classification"),
    )


# ─── Error helper ─────────────────────────────────────────────────────────────

def _error(message: str) -> dict:
    return {
        "status": "error",
        "failed_at": AGENT_NAME,
        "error": {
            "technical": message,
            "accessible": (
                "O sistema não conseguiu determinar o objetivo da análise com segurança. "
                f"Detalhe: {message}"
            ),
        },
    }
