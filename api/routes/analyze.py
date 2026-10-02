"""
POST /api/v1/analyze
Spec: seção 3.1 (entrada) e seção 3.2 (saída)

Recebe dataset + prompt, executa o pipeline AgenticML e retorna o output do Agente 6.
Quando status="done", inclui `pipeline_artifact` (base64) para uso no /predict.
"""

import base64
import io
from typing import Optional

import joblib
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core.config import settings
from core.graph import build_graph
from core.state import initial_state

router = APIRouter()

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
ALLOWED_PROBLEM_TYPES = {"classification", "regression", "clustering", "time_series", "anomaly_detection"}


@router.post("/analyze", summary="Executar pipeline AgenticML")
async def analyze(
    dataset: UploadFile = File(..., description="Arquivo CSV, XLSX ou XLS (máx 50MB / 500k linhas)"),
    prompt: str = Form(..., description="Objetivo da análise em linguagem natural"),
    problem_type: Optional[str] = Form(
        None,
        description="Sugestão de tipo de problema: classification | regression | clustering | time_series | anomaly_detection",
    ),
    random_state: int = Form(
        42,
        description="Semente de aleatoriedade para reprodutibilidade (padrão: 42).",
    ),
    balance_strategy: str = Form(
        "auto",
        description="Tratamento de desbalanceamento de classes: auto | none | class_weight (padrão: auto).",
    ),
) -> JSONResponse:
    """
    Recebe o dataset e o prompt do usuário, executa o pipeline de 6 agentes
    e retorna o output gerado pelo Agente 6 (explicação + previsões).
    """
    filename = dataset.filename or "dataset"

    # Valida extensão antes de ler os bytes
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado: '.{ext}'. Use .csv, .xlsx ou .xls.",
        )

    # Valida problem_type se fornecido
    if problem_type and problem_type not in ALLOWED_PROBLEM_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de problema inválido: '{problem_type}'. Opções: {', '.join(ALLOWED_PROBLEM_TYPES)}.",
        )

    if balance_strategy not in {"auto", "none", "class_weight"}:
        raise HTTPException(
            status_code=400,
            detail=f"balance_strategy inválido: '{balance_strategy}'. Opções: auto, none, class_weight.",
        )

    raw_bytes = await dataset.read()

    # Valida tamanho do arquivo
    if len(raw_bytes) > settings.max_file_size_bytes:
        size_mb = len(raw_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo ({size_mb:.1f}MB) excede o limite de {settings.MAX_FILE_SIZE_MB}MB.",
        )

    # Monta o estado inicial e executa o grafo
    state = initial_state(
        dataset_bytes=raw_bytes,
        filename=filename,
        prompt=prompt,
        problem_type_hint=problem_type,
        random_state=random_state,
        balance_strategy=balance_strategy,
    )

    graph = build_graph()
    final_state = graph.invoke(state)

    output = final_state.get("output") or {
        "status": "error",
        "explanation": {
            "technical": "Pipeline concluído sem output. Verifique os logs.",
            "accessible": "Ocorreu um erro inesperado. Tente novamente.",
        },
    }

    # Inclui artefato de pipeline para uso posterior em /predict (spec 6.6)
    if output.get("status") == "done":
        training = final_state.get("training") or {}
        features = final_state.get("features") or {}
        interpretation = final_state.get("interpretation") or {}

        if training.get("model_artifact"):
            pipeline_bundle = {
                "model_artifact": training["model_artifact"],
                "fitted_params": features.get("fitted_params", {}),
                "feature_columns": features.get("feature_columns", []),
                "problem_type": interpretation.get("problem_type", "classification"),
                "label_encoder_target": training.get("label_encoder_target"),
            }
            buf = io.BytesIO()
            joblib.dump(pipeline_bundle, buf)
            output["pipeline_artifact"] = base64.b64encode(buf.getvalue()).decode()

    status_code = 200 if output.get("status") != "error" else 422
    return JSONResponse(content=output, status_code=status_code)
