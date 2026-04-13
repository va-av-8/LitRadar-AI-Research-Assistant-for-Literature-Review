"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_BASE_URL")
    semantic_scholar_api_key: str = Field(default="", alias="SEMANTIC_SCHOLAR_API_KEY")
    semantic_scholar_enabled: bool = Field(default=True, alias="SEMANTIC_SCHOLAR_ENABLED")

    # Model settings
    llm_model_default: str = Field(default="openai/gpt-4o-mini", alias="LLM_MODEL_DEFAULT")
    llm_model_synthesis: str = Field(default="openai/gpt-4o-mini", alias="LLM_MODEL_SYNTHESIS")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")

    # Limits and SLO
    max_iterations: int = Field(default=2, alias="MAX_ITERATIONS")
    max_papers_per_session: int = Field(default=30, alias="MAX_PAPERS_PER_SESSION")
    search_results_per_query: int = Field(default=10, alias="SEARCH_RESULTS_PER_QUERY")
    critic_batch_size: int = Field(default=10, alias="CRITIC_BATCH_SIZE")
    kb_lookup_top_k: int = Field(default=10, alias="KB_LOOKUP_TOP_K")
    retriever_top_k_synthesis: int = Field(default=5, alias="RETRIEVER_TOP_K_SYNTHESIS")
    chroma_db_path: str = Field(default="./chroma_db", alias="CHROMA_DB_PATH")
    api_timeout_seconds: int = Field(default=10, alias="API_TIMEOUT_SECONDS")
    api_retry_count: int = Field(default=3, alias="API_RETRY_COUNT")
    budget_soft_limit_usd: float = Field(default=0.08, alias="BUDGET_SOFT_LIMIT_USD")
    budget_hard_limit_usd: float = Field(default=0.15, alias="BUDGET_HARD_LIMIT_USD")
    injection_rate_stop_threshold: float = Field(default=0.30, alias="INJECTION_RATE_STOP_THRESHOLD")

    # Langfuse
    langfuse_tracing: bool = Field(default=True, alias="LANGFUSE_TRACING")
    langfuse_project: str = Field(default="litradar-poc", alias="LANGFUSE_PROJECT")

    # Topic anchors for Input Guard
    topic_anchors: list[str] = Field(
        default=[
            "natural language processing transformer",
            "computer vision image recognition",
            "reinforcement learning policy gradient",
            "large language model reasoning",
            "machine learning optimization",
        ]
    )

    # Topic similarity threshold
    topic_similarity_threshold: float = Field(default=0.35)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
