# AgenticML

Sistema de Machine Learning Automatizado e Explicável orquestrado por agentes baseados em LLMs.

**TCC — MBA em Data Science e Analytics**  
Autor: Rafael Ponte Gutierres

---

## Visão Geral

O AgenticML recebe um dataset e um prompt em linguagem natural e retorna um pipeline de ML treinado com explicações. O sistema é composto por 6 agentes LangGraph que executam sequencialmente:

| # | Agente | Responsabilidade |
|---|--------|-----------------|
| 1 | Ingestão e Inspeção | Leitura, parsing e inspeção estrutural |
| 2 | Diagnóstico de Qualidade | Missing values, desbalanceamento, leakage |
| 3 | Feature Engineering | Transformações com justificativas em PT-BR |
| 4 | Interpretação NL | Inferência do tipo de problema via LLM |
| 5 | Seleção e Treinamento | Comparação de candidatos e treinamento |
| 6 | Previsão e Explicação | Relatório narrativo técnico e acessível |

---

## Stack

- **Python 3.11**
- **LangGraph + LangChain** — orquestração dos agentes
- **GPT-4o (OpenAI) / Claude Sonnet (Anthropic)** — LLMs
- **FastAPI** — API REST
- **scikit-learn + XGBoost** — treinamento de modelos

---

## Instalação

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**macOS:** XGBoost requer `brew install libomp`.

Configure as chaves de API em `.env` (copie de `.env.example`):

```bash
cp .env.example .env
# Edite .env com suas chaves
```

---

## Uso

### API REST

```bash
uvicorn main:app --reload
```

**POST /api/v1/analyze** — executa o pipeline completo:
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "dataset=@meu_dataset.csv" \
  -F "prompt=Prever se o cliente vai cancelar o contrato"
```

**POST /api/v1/predict** — previsões com modelo treinado:
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "prediction_data=@novos_dados.csv" \
  -F "pipeline_artifact=@pipeline.joblib"
```

O campo `pipeline_artifact` da resposta do `/analyze` deve ser decodificado de base64 e salvo como `.joblib` antes de enviar ao `/predict`.

### Notebooks (TCC)

```bash
jupyter lab
```

Abra os notebooks em `notebooks/experiments/`:

| Notebook | Dataset | Tipo |
|----------|---------|------|
| `01_iris_classification.ipynb` | Iris | Classificação multiclasse |
| `02_titanic_classification.ipynb` | Titanic | Classificação binária |
| `03_california_housing_regression.ipynb` | California Housing | Regressão |

---

## Modo Fallback

Todos os agentes possuem fallback determinístico quando a API key não está configurada. O sistema é completamente funcional sem LLM — as explicações e justificativas são geradas por templates.

---

## Spec

A especificação completa do sistema está em [`agenticml_spec.md`](agenticml_spec.md).  
Em qualquer conflito entre instrução e spec, **a spec prevalece**.
