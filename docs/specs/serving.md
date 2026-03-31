# Spec: Serving & Config

---

## Запуск

```bash
# Установка uv (если не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Создание окружения и установка зависимостей
uv sync

# Запуск
uv run streamlit run app.py
```

Приложение доступно на `http://localhost:8501`. Облачный деплой вне scope PoC.

> **Docker** — не используется в PoC. Добавить в будущем при необходимости деплоя.

---

## Конфигурация

Все параметры читаются из переменных окружения через `python-dotenv` (файл `.env`).

### Секреты (не коммитить в git)

```env
OPENROUTER_API_KEY=sk-or-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # или self-hosted URL
SEMANTIC_SCHOLAR_API_KEY=                  # опционально, без ключа работает с лимитом 100 req/min
```

### Параметры модели

```env
LLM_MODEL_DEFAULT=google/gemini-2.0-flash
LLM_MODEL_SYNTHESIS=google/gemini-2.0-flash   # можно заменить на openai/gpt-4.1
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Лимиты и SLO

```env
MAX_ITERATIONS=2
MAX_PAPERS_PER_SESSION=30
SEARCH_RESULTS_PER_QUERY=10
CRITIC_BATCH_SIZE=10
KB_LOOKUP_TOP_K=10
RETRIEVER_TOP_K_SYNTHESIS=5
CHROMA_DB_PATH=./chroma_db
API_TIMEOUT_SECONDS=10
API_RETRY_COUNT=3
BUDGET_SOFT_LIMIT_USD=0.08
BUDGET_HARD_LIMIT_USD=0.15
INJECTION_RATE_STOP_THRESHOLD=0.30
```

### Langfuse

```env
LANGFUSE_TRACING=true
LANGFUSE_PROJECT=litradar-poc
```

---

## Структура проекта

```
litradar/
├── app.py                  # Streamlit entry point
├── pyproject.toml          # зависимости и метаданные проекта (uv)
├── uv.lock                 # lock-файл (коммитить в git)
├── .env                    # секреты (в .gitignore)
├── .env.example            # шаблон без значений
├── chroma_db/              # персистентная ChromaDB (в .gitignore)
├── src/
│   ├── orchestrator.py     # LangGraph StateGraph, routing functions
│   ├── state.py            # AgentState TypedDict
│   ├── agents/
│   │   ├── planner.py
│   │   ├── critic.py
│   │   ├── synthesis.py
│   │   └── renderer.py
│   ├── search/
│   │   ├── arxiv_client.py
│   │   ├── semantic_scholar_client.py
│   │   └── sanitizer.py    # anti-injection layer
│   ├── retriever.py        # ChromaDB wrapper (kb_lookup, index, retrieve)
│   ├── verifier.py
│   └── config.py           # pydantic Settings из env
├── docs/
│   ├── system-design.md
│   ├── diagrams/
│   └── specs/
├── evals/
│   ├── test_topics.json    # 10 тестовых тем с эталонными списками
│   └── eval_runner.py
└── logs/                   # локальные WARNING/ERROR логи
    └── .gitkeep
```

---

## Зависимости (pyproject.toml)

```toml
[project]
name = "litradar"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "openai>=1.0.0",          # OpenRouter-совместимый клиент для LLM
    "sentence-transformers>=2.7.0",  # локальные эмбеддинги
    "chromadb>=0.5.0",
    "arxiv>=2.1.0",
    "httpx>=0.27.0",
    "streamlit>=1.40.0",
    "langfuse>=2.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
]
```
