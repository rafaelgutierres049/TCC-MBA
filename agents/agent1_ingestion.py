"""
Agente 1 — Ingestão e Inspeção
Spec: seção 6.1

Responsabilidade: Receber o dataset, identificar sua estrutura (colunas, tipos de
variáveis, distribuições, valores ausentes) e popular o State para os agentes seguintes.

Invariantes (spec 6.1):
  - Dataset vazio, corrompido ou ilegível → curto-circuito via estado de erro
  - Nunca passa adiante um dataset com 0 linhas válidas
  - Dataset com uma única coluna → curto-circuito (edge case 1, spec 8)

Segurança (edge case 7, spec 8):
  - Conteúdo das células é sanitizado antes de ser armazenado no State
  - Nenhum conteúdo do dataset é interpretado como instrução pelo LLM
"""

import csv
import io
import math
from typing import Any

import chardet
import pandas as pd

from core.config import settings
from core.state import AgenticMLState

AGENT_NAME = "Agente 1 — Ingestão e Inspeção"
SAMPLE_ROWS = 5
SUPPORTED_EXTENSIONS = {"csv", "xlsx", "xls"}


# ---------------------------------------------------------------------------
# Ponto de entrada do nó LangGraph
# ---------------------------------------------------------------------------

def run_agent1(state: AgenticMLState) -> dict:
    """
    Nó LangGraph do Agente 1.
    Recebe o estado com dataset em bytes e retorna atualização parcial do State.
    """
    raw_bytes: bytes = state["dataset"]
    filename: str = state.get("filename", "dataset.csv")  # type: ignore[call-overload]

    # 1. Valida extensão
    ext = _get_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        return _error(
            f"Formato de arquivo não suportado: '.{ext}'. "
            f"Formatos aceitos: {', '.join('.' + e for e in SUPPORTED_EXTENSIONS)}."
        )

    # 2. Valida tamanho
    if len(raw_bytes) == 0:
        return _error("O arquivo enviado está vazio.")

    if len(raw_bytes) > settings.max_file_size_bytes:
        size_mb = len(raw_bytes) / (1024 * 1024)
        return _error(
            f"O arquivo tem {size_mb:.1f}MB e excede o limite de {settings.MAX_FILE_SIZE_MB}MB."
        )

    # 3. Detecta encoding
    encoding = _detect_encoding(raw_bytes)

    # 4. Carrega o dataset
    try:
        df, separator = _load_dataset(raw_bytes, ext, encoding)
    except Exception as exc:
        return _error(f"Falha ao ler o arquivo: {exc}")

    # 5. Invariante: dataset não pode ter 0 linhas válidas
    if df is None or df.empty:
        return _error("O dataset está vazio — nenhuma linha válida foi encontrada.")

    if len(df) == 0:
        return _error("O dataset não contém linhas válidas após o parsing.")

    # 6. Invariante / Edge case 1: mínimo de 2 colunas para ML
    if len(df.columns) <= 1:
        n = len(df.columns)
        cols = list(df.columns)
        return _error(
            f"O dataset possui apenas {n} coluna(s): {cols}. "
            "São necessárias pelo menos 2 colunas para análise de Machine Learning."
        )

    # 7. Valida limite de linhas
    if len(df) > settings.MAX_ROWS:
        return _error(
            f"O dataset possui {len(df):,} linhas, excedendo o limite de "
            f"{settings.MAX_ROWS:,} linhas suportadas pelo sistema."
        )

    # 8. Gera o output de inspeção (spec 6.1 — formato de saída)
    inspection = _build_inspection(df, encoding, separator)

    return {
        "dataset": df,          # substitui bytes pelo DataFrame para agentes seguintes
        "inspection": inspection,
        "status": "running",
    }


# ---------------------------------------------------------------------------
# Detecção de encoding e separador
# ---------------------------------------------------------------------------

def _detect_encoding(raw_bytes: bytes) -> str:
    """Detecta encoding via chardet; fallback para utf-8."""
    sample = raw_bytes[:100_000]  # amostra para velocidade
    result = chardet.detect(sample)
    detected = result.get("encoding") or "utf-8"
    # Normaliza variantes do utf-8
    if detected.lower() in ("ascii", "utf-8-sig"):
        detected = "utf-8"
    return detected


def _detect_separator(raw_bytes: bytes, encoding: str) -> str:
    """Detecta o separador CSV via csv.Sniffer; fallback para vírgula."""
    try:
        sample = raw_bytes[:8192].decode(encoding, errors="replace")
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","


def _get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"


# ---------------------------------------------------------------------------
# Carregamento do dataset
# ---------------------------------------------------------------------------

def _load_dataset(
    raw_bytes: bytes, ext: str, encoding: str
) -> tuple[pd.DataFrame, str | None]:
    """Carrega o dataset de acordo com a extensão detectada."""
    if ext == "csv":
        separator = _detect_separator(raw_bytes, encoding)
        try:
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=encoding,
                sep=separator,
                low_memory=False,
            )
        except UnicodeDecodeError:
            # Fallback Latin-1 (spec 3.1: suporta UTF-8 e Latin-1)
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding="latin-1",
                sep=separator,
                low_memory=False,
            )
        return df, separator

    elif ext == "xlsx":
        df = pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")
        return df, None

    elif ext == "xls":
        df = pd.read_excel(io.BytesIO(raw_bytes), engine="xlrd")
        return df, None

    raise ValueError(f"Extensão não tratada: {ext}")


# ---------------------------------------------------------------------------
# Construção do output de inspeção (spec 6.1 — formato de saída)
# ---------------------------------------------------------------------------

def _build_inspection(
    df: pd.DataFrame, encoding: str, separator: str | None
) -> dict:
    schema = {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": list(df.shape),
    }

    descriptive_stats: dict[str, dict] = {}
    for col in df.columns:
        col_data = df[col]
        stats: dict[str, Any] = {"nulls": int(col_data.isna().sum())}

        if pd.api.types.is_numeric_dtype(col_data):
            stats["mean"] = _safe_float(col_data.mean())
            stats["std"] = _safe_float(col_data.std())
            stats["min"] = _safe_float(col_data.min())
            stats["max"] = _safe_float(col_data.max())
            stats["median"] = _safe_float(col_data.median())
        else:
            top = col_data.value_counts().head(5)
            stats["unique_count"] = int(col_data.nunique())
            # sanitiza top values antes de armazenar no State (edge case 7)
            stats["top_values"] = {
                _sanitize_cell(str(k)): int(v) for k, v in top.items()
            }

        descriptive_stats[col] = stats

    # Amostra sanitizada — conteúdo do dataset nunca interpretado como instrução (spec 8 #7)
    sample = _sanitize_sample(df.head(SAMPLE_ROWS))

    return {
        "schema": schema,
        "descriptive_stats": descriptive_stats,
        "sample": sample,
        "encoding_detected": encoding,
        "separator_detected": separator if separator is not None else "N/A (Excel)",
    }


# ---------------------------------------------------------------------------
# Sanitização contra prompt injection (spec edge case 7)
# ---------------------------------------------------------------------------

def _sanitize_cell(value: str) -> str:
    """
    Remove/escapa sequências que poderiam ser interpretadas como instruções pelo LLM.
    Aplicado a todo conteúdo do dataset antes de armazenar no State.
    """
    # Remove quebras de linha que criariam contexto de multi-linha no prompt
    value = value.replace("\n", " ").replace("\r", " ")
    # Escapa delimitadores de bloco de código
    value = value.replace("```", "'''")
    # Remove tags que poderiam ser confundidas com instruções de sistema
    value = value.replace("<|", "< |").replace("|>", "| >")
    return value.strip()


def _sanitize_sample(df: pd.DataFrame) -> list[list[str]]:
    """Converte amostra do DataFrame em lista sanitizada para armazenamento no State."""
    sanitized = []
    for _, row in df.iterrows():
        sanitized_row = [_sanitize_cell(str(v)) for v in row]
        sanitized.append(sanitized_row)
    return sanitized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    """Converte para float tratando NaN/Inf (não serializáveis em JSON)."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _error(message: str) -> dict:
    """
    Constrói o estado de curto-circuito (spec 7.2).
    O Agente 6 em modo erro usa este payload para gerar a resposta ao usuário.
    """
    return {
        "status": "error",
        "failed_at": AGENT_NAME,
        "error": {
            "technical": message,
            "accessible": (
                "O sistema não conseguiu processar o arquivo enviado. "
                f"Motivo: {message}"
            ),
        },
    }
