# Spec: Memory & Context

---

## Session State (AgentState)

Единственное хранилище состояния сессии — `AgentState` TypedDict в LangGraph. Передаётся между всеми нодами графа. Не персистируется между сессиями.

```python
from typing import TypedDict, Optional, Set, List

class PaperMetadata(TypedDict):
    paper_id: str          # "arxiv:..." или "s2:..."
    title: str
    authors: List[str]
    abstract: Optional[str]
    year: Optional[int]
    source: str            # "arxiv" | "semantic_scholar"
    citation_count: Optional[int]

class Contradiction(TypedDict):
    topic: str
    paper_a: str           # paper_id
    paper_b: str           # paper_id
    description: str

class Citation(TypedDict):
    paper_id: str
    claim: str             # конкретное утверждение, которое цитирует статью

class SynthesisJSON(TypedDict):
    sections: List[dict]   # [{title, summary, paper_ids}]
    citations: List[Citation]
    detected_contradictions: List[Contradiction]
    open_questions: List[str]

class AgentState(TypedDict):
    # Входные данные
    query: str

    # Управление итерациями
    iteration: int                          # текущая итерация, начиная с 0
    max_iterations: int                     # = 2 (из config)
    prev_open_questions: List[str]          # для детекта зависания loop

    # Планирование
    subqueries: List[str]

    # База знаний (из персистентной ChromaDB, начало сессии)
    papers_from_kb: List[PaperMetadata]

    # Поиск
    papers_seen: Set[str]                   # все paper_id за сессию (дедупликация)
    papers_found: List[PaperMetadata]       # все статьи от Search Agent
    papers_accepted: List[PaperMetadata]    # после Critic (только новые)

    # Синтез
    synthesis_output: Optional[SynthesisJSON]  # содержит sections, citations, detected_contradictions, open_questions

    # Верификация
    verified_citations: List[Citation]
    removed_citations: List[Citation]

    # Финал
    final_review: Optional[str]            # Markdown от Renderer

    # Инфраструктура
    sources_available: List[str]           # ["arxiv", "semantic_scholar"]
    token_count: int                       # суммарные токены сессии
    session_budget_remaining: float        # остаток в $
    has_injection_signal: bool
    injection_rate: float                  # доля абстрактов с injection-паттернами
    source_gap_note: Optional[str]         # пометка об отсутствующем источнике
    error: Optional[str]                   # сообщение об ошибке для UI
```

---

## Memory Policy

| Тип памяти | Реализация | Персистентность |
|---|---|---|
| Session state | LangGraph AgentState | Только в рамках сессии |
| База знаний (векторный индекс) | ChromaDB, на диске (`./chroma_db/`) | Между сессиями, накапливается |
| LLM traces | Langfuse | 30–90 дней (по типу события) |
| Локальные логи | JSON-файл | 90–180 дней (по типу события) |

История пользовательских запросов между сессиями **не сохраняется**. ChromaDB хранит только проиндексированные статьи, не запросы.

---

## Context Budget (управление контекстным окном)

### Принцип

Ни один агент не получает весь корпус абстрактов в контексте. Critic оценивает `papers_found` батчами из AgentState; Synthesis использует RAG retrieval из ChromaDB.

### Лимиты по агентам

| Агент | Контекст | Примерный размер |
|---|---|---|
| Planner | query + заголовки papers_from_kb + open_questions (iter 2) | ~800 токенов |
| Critic | системный промпт + батч ~10 абстрактов из AgentState | ~3 000–4 000 токенов |
| Synthesis | системный промпт + top-5 абстрактов per cluster × N clusters | ~5 000–6 000 токенов |
| Renderer | системный промпт + верифицированный synthesis JSON | ~2 000–3 000 токенов |

### Контроль бюджета

- `token_count` обновляется после каждого LLM-вызова (из response.usage)
- `session_budget_remaining` = `BUDGET_HARD_LIMIT - (token_count * price_per_token)`
- Budget Guard проверяется **перед каждым** LLM-вызовом в Orchestrator

---

## Обработка абстрактов

Все абстракты оборачиваются в тег `<external_content>` и проходят через Sanitization Layer до передачи любому агенту. Детали реализации — в `specs/agent-orchestrator.md` (раздел Sanitization Layer).
