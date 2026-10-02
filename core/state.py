from typing import Any, TypedDict


class AgenticMLState(TypedDict):
    """
    Estado compartilhado entre todos os agentes do pipeline LangGraph.
    Fonte da verdade: spec seção 7.4.
    Nenhum agente acessa o output de outro diretamente — toda comunicação é via este State.
    """

    # --- Input ---
    dataset: Any              # bytes → pd.DataFrame após Agent 1
    filename: str             # nome original do arquivo (para detectar extensão)
    prompt: str               # objetivo do usuário em linguagem natural
    problem_type_hint: str | None  # sugestão do tipo de problema (spec 3.1)
    prediction_data: Any      # dados para previsão (opcional, spec 6.6)
    random_state: int         # semente de aleatoriedade (reprodutibilidade / repetições multi-seed)
    balance_strategy: str     # "auto" | "none" | "class_weight" — tratamento de desbalanceamento (Agente 5)

    # --- Outputs por agente ---
    inspection: dict | None      # Agente 1 — Ingestão e Inspeção
    quality: dict | None         # Agente 2 — Diagnóstico de Qualidade
    features: dict | None        # Agente 3 — Feature Engineering
    interpretation: dict | None  # Agente 4 — Interpretação de Linguagem Natural
    training: dict | None        # Agente 5 — Seleção e Treinamento
    output: dict | None          # Agente 6 — Previsão e Explicação

    # --- Controle de fluxo ---
    status: str        # "running" | "error" | "done"
    error: dict | None 
    failed_at: str | None  # nome do agente que falhou


def initial_state(
    dataset_bytes: bytes,
    filename: str,
    prompt: str,
    problem_type_hint: str | None = None,
    prediction_data: Any = None,
    random_state: int = 42,
    balance_strategy: str = "auto",
) -> AgenticMLState:
    """Factory para criar o estado inicial com todos os campos obrigatórios."""
    return AgenticMLState(
        dataset=dataset_bytes,
        filename=filename,
        prompt=prompt,
        problem_type_hint=problem_type_hint,
        prediction_data=prediction_data,
        random_state=random_state,
        balance_strategy=balance_strategy,
        inspection=None,
        quality=None,
        features=None,
        interpretation=None,
        training=None,
        output=None,
        status="running",
        error=None,
        failed_at=None,
    )
