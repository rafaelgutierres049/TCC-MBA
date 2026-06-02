"""
AgenticML — Entry point FastAPI
Spec: seção 4 (Interfaces de Uso)

Execução:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analyze import router as analyze_router
from api.routes.predict import router as predict_router

app = FastAPI(
    title="AgenticML",
    description=(
        "Sistema de Machine Learning Automatizado e Explicável "
        "orquestrado por agentes baseados em LLMs.\n\n"
        "TCC — MBA em Data Science e Analytics."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/api/v1", tags=["Pipeline"])
app.include_router(predict_router, prefix="/api/v1", tags=["Previsão"])


@app.get("/health", tags=["Sistema"])
def health_check():
    """Verifica se a API está respondendo."""
    return {"status": "ok", "version": "0.1.0"}
