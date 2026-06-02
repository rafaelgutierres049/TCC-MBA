"""
Factory centralizada de LLM.
Evita duplicação do código de instanciação nos agentes.
Todos os agentes que precisam de LLM importam get_llm() daqui.
"""

from core.config import settings


def get_llm():
    """Retorna a instância de LLM configurada via PRIMARY_LLM no .env."""
    if settings.PRIMARY_LLM == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            api_key=settings.ANTHROPIC_API_KEY,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        api_key=settings.OPENAI_API_KEY,
    )
