from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLMs ---
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5"
    PRIMARY_LLM: str = "openai"  # "openai" | "anthropic"

    # --- Limites do pipeline ---
    MAX_FILE_SIZE_MB: int = 50
    MAX_ROWS: int = 500_000
    MAX_ROWS_PREVIEW: int = 10_000

    # --- LLM runtime ---
    LLM_TEMPERATURE: float = 0.0 
    LLM_MAX_TOKENS: int = 4096

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


settings = Settings()
