"""
Centralized settings, loaded from environment variables (see .env.example).
Every other module imports `settings` from here instead of calling
os.environ directly, so configuration stays in one place.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = "./chroma_db"

    # Monitoring
    metric_sample_interval_seconds: int = 5
    metric_history_length: int = 720  # keep last N samples per metric

    # Anomaly detection
    anomaly_contamination: float = 0.05

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:8080"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
