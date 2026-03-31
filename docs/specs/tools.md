# Spec: Tools & API Integrations

---

## 1. ArXiv API

### Контракт

- Библиотека: `arxiv` (PyPI)
- Метод: `arxiv.Search(query, max_results, sort_by=arxiv.SortCriterion.Relevance)`
- Возвращает: `id`, `title`, `authors`, `summary` (abstract), `published`, `categories`

### Нормализация

```python
{
    "paper_id": "arxiv:{entry_id}",   # e.g. "arxiv:2301.07597"
    "title": str,
    "authors": List[str],
    "abstract": str,
    "year": int,                       # из published.year
    "source": "arxiv",
    "citation_count": None             # ArXiv не предоставляет
}
```

### Ограничения и защиты

| Параметр | Значение |
|---|---|
| Rate limit | 3 req/sec, без ключа |
| Timeout | 10 сек |
| Retry | 3 попытки, exponential backoff (1s, 2s, 4s) |
| max_results per query | 10 |
| Side effects | Нет — read-only |

### Обработка ошибок

- HTTP 429 → backoff + retry
- Timeout → retry, после 3 попыток: `sources_available.remove("arxiv")`, продолжить с Semantic Scholar

---

## 2. Semantic Scholar API

### Контракт

- Endpoint: `GET https://api.semanticscholar.org/graph/v1/paper/search`
- Params: `query`, `fields=paperId,title,authors,abstract,year,citationCount`, `limit=10`
- Auth: без ключа (100 req/min), с ключом (выше лимиты)

### Нормализация

```python
{
    "paper_id": "s2:{paperId}",        # e.g. "s2:204e3073870fae3d05bcbc2f6a8e263d28be9e82"
    "title": str,
    "authors": List[str],
    "abstract": str | None,            # может отсутствовать
    "year": int | None,
    "source": "semantic_scholar",
    "citation_count": int | None
}
```

### Ограничения и защиты

| Параметр | Значение |
|---|---|
| Rate limit | 1 req/sec рекомендуется (100 req/min без ключа) |
| Timeout | 10 сек |
| Retry | 3 попытки, exponential backoff |
| Side effects | Нет — read-only |

### Обработка ошибок

- HTTP 429 → backoff + retry
- Отсутствующий abstract → индексировать по `title + authors`
- После 3 неудачных попыток → `sources_available.remove("semantic_scholar")`

---

## 3. Verifier HTTP-проверки

### ArXiv верификация

```
GET https://export.arxiv.org/abs/{arxiv_id}
Успех: HTTP 200
Провал: HTTP 404, 5xx, timeout
```

### Semantic Scholar верификация

```
GET https://api.semanticscholar.org/graph/v1/paper/{paper_id}
Успех: HTTP 200 + совпадение paperId в ответе
Провал: HTTP 404, 5xx, timeout
```

### Политика верификации

- Timeout: 5 сек
- Retry: 1 попытка
- Провал верификации → цитата удаляется из `verified_citations`, добавляется в `removed_citations`
- `removed_citations` логируются как WARNING
- Если все цитаты не прошли верификацию → ошибка, сообщение пользователю

---

## 4. OpenRouter API (LLM)

### Контракт

- Endpoint: `https://openrouter.ai/api/v1/chat/completions` (OpenAI-совместимый)
- Модель (default): `google/gemini-2.0-flash`
- Модель (Synthesis, опционально): `openai/gpt-4.1`
- Все вызовы: structured output через JSON mode
- Трейсинг: Langfuse callback на каждый вызов

### Бюджетный контроль

| Порог | Действие |
|---|---|
| token_count → $0.08 | WARNING в Langfuse, предупреждение в UI |
| token_count → $0.15 | Hard stop: принудительное завершение графа |

### Обработка ошибок

- HTTP 529 (overloaded) → retry с backoff
- HTTP 400 (invalid request) → логировать как ERROR, остановка сессии
- Timeout 30 сек → retry 1 раз, затем ERROR

---

## 6. Embeddings (локально)

### Назначение

Эмбеддинги генерируются локально через `sentence-transformers` — внешние API не требуются, стоимость нулевая.

### Модель

- `BAAI/bge-small-en-v1.5` (768 dim, ~130MB)
- Скачивается автоматически при первом запуске через HuggingFace Hub
- Скорость на CPU: ~10–30ms на абстракт; 20 абстрактов < 1 сек

### Точки вызова

- **Input Guard**: эмбеддинг запроса для тематической фильтрации (cosine similarity с якорями)
- **Retriever**: эмбеддинг при индексировании `papers_accepted` и при каждом retrieval-запросе

### Зависимость

`sentence-transformers>=2.7.0` в `pyproject.toml`. Отдельный API-ключ не нужен.

---

## 7. Langfuse (Observability)

### Интеграция

- Langfuse SDK передаётся в каждый LLM-вызов через callback
- Трейс создаётся per-session, включает все ноды LangGraph
- Ключи: `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` из env

### Side effects

- Трейсы отправляются асинхронно — не блокируют основной поток
- При недоступности Langfuse: логировать локально в файл, продолжить работу
