"""
POST /api/v1/predict
Spec: seção 6.6 (modos de previsão — requisição separada)

Fluxo:
  1. Cliente chama /analyze → recebe pipeline_artifact (base64 de bundle joblib)
  2. Cliente chama /predict com pipeline_artifact + novos dados → recebe previsões

O bundle contém: model_artifact, fitted_params, feature_columns,
                 problem_type, label_encoder_target.

Edge case 5 (spec 8): schema de prediction_data diferente do treino → 422.
"""

import base64
import io

import joblib
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from agents.agent6_explanation import _SchemaError, _generate_predictions

router = APIRouter()

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}


@router.post("/predict", summary="Gerar previsões com pipeline treinado")
async def predict(
    prediction_data: UploadFile = File(
        ..., description="Novos dados para previsão (CSV, XLSX ou XLS)"
    ),
    pipeline_artifact: UploadFile = File(
        ..., description="Artefato de pipeline retornado pelo /analyze (arquivo .joblib)"
    ),
) -> JSONResponse:
    """
    Recebe o artefato de pipeline gerado pelo /analyze e novos dados,
    aplica as mesmas transformações do treino e retorna previsões.

    O `pipeline_artifact` é o arquivo .joblib obtido decodificando o campo
    `pipeline_artifact` (base64) da resposta do /analyze.
    """
    # Valida extensão dos dados de previsão
    pred_filename = prediction_data.filename or "data.csv"
    ext = pred_filename.rsplit(".", 1)[-1].lower() if "." in pred_filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado: '.{ext}'. Use .csv, .xlsx ou .xls.",
        )

    # Lê bytes do pipeline e dos novos dados
    pipeline_bytes = await pipeline_artifact.read()
    pred_bytes = await prediction_data.read()

    # Desserializa o bundle
    try:
        bundle: dict = joblib.load(io.BytesIO(pipeline_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Artefato de pipeline inválido ou corrompido: {exc}",
        )

    required_keys = {"model_artifact", "fitted_params", "feature_columns"}
    if not required_keys.issubset(bundle.keys()):
        raise HTTPException(
            status_code=400,
            detail=(
                "Artefato de pipeline incompleto. "
                f"Chaves esperadas: {required_keys}. "
                f"Chaves encontradas: {set(bundle.keys())}."
            ),
        )

    # Monta dicts no formato esperado por _generate_predictions
    features_dict = {
        "fitted_params": bundle["fitted_params"],
        "feature_columns": bundle["feature_columns"],
    }
    training_dict = {
        "model_artifact": bundle["model_artifact"],
        "label_encoder_target": bundle.get("label_encoder_target"),
    }

    # Gera previsões reutilizando a lógica do Agente 6
    try:
        result = _generate_predictions(
            pred_bytes,
            pred_filename,
            features_dict,
            training_dict,
        )
    except _SchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao gerar previsões: {exc}",
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "done",
            "problem_type": bundle.get("problem_type"),
            "predictions": result,
        },
    )
