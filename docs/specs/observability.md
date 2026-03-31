# Spec: Observability & Evals

---

## Трейсинг (Langfuse)

Каждая сессия создаёт один трейс в Langfuse с вложенными спанами по нодам LangGraph.

### Структура трейса

```
session:{session_id}
  ├── input_guard
  ├── kb_lookup [papers_from_kb=5]
  ├── plan [iteration=0]
  ├── search [sources=arxiv,semantic_scholar]
  ├── sanitize [injection_signals=0]
  ├── critique [accepted=18, rejected=4]
  ├── index [papers_indexed=18]
  ├── synthesize [contradictions=2, open_questions=1]
  ├── plan [iteration=1]          ← если reflection loop
  ├── search [iteration=1]
  ├── ...
  ├── citation_guard [removed=1]
  ├── verify [verified=16, removed=2]
  └── render
```

### Метрики, собираемые per-session

| Метрика | Тип | Описание |
|---|---|---|
| `total_latency_ms` | gauge | End-to-end время сессии |
| `llm_calls_count` | counter | Число LLM-вызовов |
| `total_tokens` | counter | Суммарный расход токенов |
| `total_cost_usd` | gauge | Суммарная стоимость |
| `papers_found` | gauge | Найдено поиском |
| `papers_accepted` | gauge | После Critic |
| `papers_rejected_rate` | gauge | papers_rejected / papers_found |
| `citations_removed_count` | gauge | Удалено Verifier'ом |
| `iterations_count` | gauge | Итераций reflection loop |
| `open_questions_resolved` | gauge | Закрытых open_questions на iter 2 |
| `contradictions_found` | gauge | Противоречий в обзоре |

---

## Логирование

Все события логируются структурированным JSON:

```json
{
  "timestamp": "2025-01-15T14:23:01Z",
  "session_id": "sess_abc123",
  "agent_name": "critic",
  "event_type": "paper_rejected",
  "payload": {
    "paper_id": "arxiv:2301.07597",
    "reason": "Not related to CoT, focuses on retrieval augmentation"
  },
  "latency_ms": 1240
}
```

### Уровни и хранение

| Уровень | Событие | Хранилище | Срок |
|---|---|---|---|
| INFO | query, subqueries, paper metadata, Critic decisions, Synthesis structure | Langfuse | 30 дней |
| WARNING | Removed citations, budget alerts, iteration limit hits | Langfuse + `logs/warnings.jsonl` | 90 дней |
| ERROR | API failures, LLM parse errors, budget hard stop | Langfuse + `logs/errors.jsonl` | 90 дней |
| WARNING | Potential injection signal | `logs/security.jsonl` отдельно | 180 дней |

### Что не логируется

- Полные тексты абстрактов (только paper_id, title, authors, year)
- Сырые LLM-ответы в production режиме (только в debug)
- Данные пользователя, кроме исходного query

---

## Evals

### Тестовый датасет

10 тем с известными ключевыми работами, размеченными вручную:

```json
[
  {
    "query": "chain-of-thought prompting in large language models",
    "expected_papers": ["arxiv:2201.11903", "arxiv:2205.01068"],
    "min_coverage": 0.7,
    "should_have_contradictions": true
  }
]
```

### Метрики качества

| Метрика | Формула | Целевое значение |
|---|---|---|
| Coverage | \|found ∩ expected\| / \|expected\| | ≥ 70% |
| Critic Precision | релевантных в accepted / всего в accepted | ≥ 80% |
| Hallucination Rate | removed_citations / total_citations | = 0% |
| Contradiction Recall | темы с противоречиями, обнаруженные / реальные | ≥ 60% |
| Open Questions Resolution | закрытых на iter 2 / открытых на iter 1 | ≥ 60% |

### Запуск evals

```bash
python evals/eval_runner.py --topics evals/test_topics.json --output evals/results/
```

Результаты сохраняются в JSON + сводная таблица в Markdown.

### Ручные проверки

- Выборочная проверка 20% финальных обзоров на релевантность
- Проверка всех `detected_contradictions` — реальные или ложные
- Проверка removed_citations — галлюцинация или ошибка Verifier

---

## Алерты

| Условие | Действие |
|---|---|
| `citation_removed_rate > 10%` в сессии | WARNING в Langfuse |
| `papers_rejected_rate > 60%` в сессии | WARNING + сигнал о деградации Planner |
| `total_cost_usd > 0.08` | Soft alert в UI |
| `total_cost_usd > 0.15` | Hard stop сессии |
| Langfuse недоступен | Fallback на локальные файлы, продолжить работу |
