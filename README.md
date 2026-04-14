# LitRadar — AI Research Assistant for Literature Review

> Агентная система для автоматизированного итеративного поиска и синтеза научной литературы по теме исследования в области ИИ/ML.

---

## Задача, аудитория и боль

**Для кого:** исследователи, аспиранты, студенты магистратуры, аналитики R&D-отделов, которые регулярно делают обзоры литературы.

**Какая боль сейчас:**
Литературный обзор по любой активной теме в ML/AI — это 5–15 часов ручной работы: поиск на ArXiv и Semantic Scholar, фильтрация нерелевантных статей, чтение абстрактов, группировка по подтемам, выявление ключевых работ и противоречий между ними. При этом человек легко пропускает важные работы или смещается в сторону знакомых авторов.

**Системная проблема:** существующие инструменты (Connected Papers, Elicit, Consensus) делают один запрос и возвращают список ссылок. Ни один из них не умеет итеративно углублять поиск на основе противоречий, найденных в промежуточных результатах.

---

## Что делает PoC на демо

1. Пользователь вводит тему исследования в свободной форме (например, *"retrieval-augmented generation for long-context reasoning"*)
2. **Planner агент** декомпозирует тему на 3–5 целевых подзапросов
3. **Search агент** ищет релевантные статьи через ArXiv API и Semantic Scholar API
4. **Critic агент** оценивает релевантность каждой найденной статьи и отсекает нерелевантные
5. **Synthesis агент** анализирует найденные статьи, обнаруживает противоречия и несогласованности между ними (multi-hop reasoning по метаданным и абстрактам) и передаёт открытые вопросы обратно в Planner
6. **Planner** при наличии открытых вопросов инициирует уточняющий поиск (reflection loop, до 2 итераций)
7. **Верификатор** проверяет, что каждая цитата в финальном обзоре реально существует в источнике
8. Финальный структурированный обзор отображается в Streamlit UI: ключевые направления, основные работы, выявленные противоречия, список источников

---

## Что НЕ делает PoC (явный out-of-scope)

| Вне scope | Почему |
|---|---|
| Полнотекстовый анализ PDF | Высокая сложность парсинга, достаточно абстрактов для PoC |
| Граф цитирований как инфраструктура (Neo4j и т.п.) | Заменяется structured state в памяти агента — паттерн сохраняется, инфраструктура нет |
| Охват за пределами AI/ML тематики | Ограничение датасета для управляемых evals |
| Мультиязычный поиск | Только английский язык |
| Персистентная память между сессиями | Следующая итерация после PoC |
| Интеграция с платными базами (IEEE, ACM, Springer) | Требует подписок и OAuth |
| Автоматическая генерация BibTeX | Вне фокуса текущего PoC |
| Telegram-бот или мобильный интерфейс | Streamlit достаточен для демо |

---

## Быстрый старт

### Онлайн-версия

Приложение доступно онлайн — не требует установки:

**https://litradar-ai-research-assistant-for-literature-re-production.up.railway.app**

Это production-версия, автоматически деплоится из ветки `main` на Railway.

### Локальная установка

```bash
# Клонирование репозитория
git clone https://github.com/your-repo/LitRadar-AI-Research-Assistant-for-Literature-Review.git
cd LitRadar-AI-Research-Assistant-for-Literature-Review

# Установка зависимостей (требуется uv)
uv sync

# Настройка переменных окружения
cp .env.example .env
# Заполнить OPENROUTER_API_KEY в .env
```

### Запуск приложения

```bash
# Streamlit UI
uv run streamlit run app.py

# Или напрямую через Python
uv run python -c "from src.orchestrator import run_pipeline; print(run_pipeline('your research topic'))"
```

### Запуск evaluation

```bash
# Полный прогон на 10 тестовых темах
uv run python evals/eval_runner.py --reset-db

# Без предзагрузки классических статей (для сравнения)
uv run python evals/eval_runner.py --reset-db --no-preload
```

---

## Деплой

Приложение контейнеризировано и готово к деплою:

- **Dockerfile** в корне репозитория (python:3.11-slim + uv)
- **Хостинг:** Railway с автодеплоем из ветки `main`
- **Секреты:** API-ключи задаются через переменные окружения на платформе (не через .env)
- **ChromaDB:** данные не персистятся между редеплоями (stateless режим)

Переменные окружения для настройки:
- `OPENROUTER_API_KEY` — обязательно
- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST` — для observability
- `SEMANTIC_SCHOLAR_API_KEY` — опционально, снимает rate limit

---

## Архитектура поиска

### Текущая реализация: классические статьи + недавние

Система использует **два источника** для балансировки между классическими и актуальными работами:

| Источник | Цель | Сортировка |
|----------|------|------------|
| **ArXiv API** | Свежие препринты | По релевантности (default) |
| **Semantic Scholar API** | Высокоцитируемые работы | По цитированиям (`citationCount:desc`) |

**Fallback на OpenAlex:** при rate limit Semantic Scholar (429) система автоматически переключается на OpenAlex API с фильтрацией по AI/ML топикам.

### Preload классических статей (evals)

Для evaluation дополнительно загружаются **top-25 высокоцитируемых статей** из OpenAlex в ChromaDB перед запуском pipeline. Это обеспечивает "фоновые знания" о классических работах.

Фильтры OpenAlex:
- `topics.id`: NLP, Neural Networks, CV
- `cited_by_count:>50`
- `publication_year:>2020`
- `indexed_in:arxiv`

### Как улучшить поиск

1. **Добавить Semantic Scholar API ключ** — снимает rate limit (100 req/5min → 1000 req/min)
2. **Расширить топики OpenAlex** — добавить reinforcement learning, optimization и др.
3. **Использовать citation graph** — искать статьи, цитирующие/цитируемые найденными (Connected Papers паттерн)
4. **Гибридная сортировка** — комбинировать relevance score и citation count
5. **Semantic search по абстрактам** — использовать embeddings для поиска похожих работ в ChromaDB

---

## Метрики evaluation

### Описание метрик

| Метрика | Формула | Цель | Что измеряет |
|---------|---------|------|--------------|
| **Retrieval Coverage** | `\|found ∩ expected\| / \|expected\|` до Critic | — | Доля эталонных статей, найденных поиском |
| **Accepted Coverage** | `\|accepted ∩ expected\| / \|expected\|` после Critic | ≥70% | Доля эталонных статей в финальном результате |
| **Hallucination Rate** | `removed_citations / (removed + verified)` | 0% | Доля цитат на несуществующие статьи |
| **Contradictions Correct** | Найдены противоречия, если ожидались | — | Бинарная проверка: нашли ли хоть одно противоречие |
| **Contradiction Recall** | `\|detected ∩ expected_pairs\| / \|expected_pairs\|` | ≥60% | Доля найденных ожидаемых пар противоречий |
| **OQ Resolution** | `\|closed_questions\| / \|open_questions\|` | ≥60% | Прогресс во второй итерации |

### Текущие результаты (2026-04-14)

**Общие метрики (10 тем):**

| Метрика | Значение | Цель |
|---------|----------|------|
| Avg Retrieval Coverage | 0% | — |
| Avg Accepted Coverage | 11.95% | ≥70% |
| Avg Hallucination Rate | 31.40% | 0% |
| Contradictions Correct | 2/10 (20%) | — |
| Avg Contradiction Recall | 0% | ≥60% |
| Avg OQ Resolution | 0% | ≥60% |
| Avg Tokens/Session | 51,364 | — |

**Что работает:**
- ✅ Pipeline стабильно завершается на всех 10 темах
- ✅ Система находит противоречия между статьями (2/10 тем)
- ✅ Verifier отсекает невалидные цитаты
- ✅ OpenAlex fallback работает при rate limit Semantic Scholar
- ✅ Preload классических статей повышает coverage

**Per-topic результаты:**

| Topic | Retr. Cov | Acc. Cov | Halluc. | Contr. Correct | Contr. Recall | OQ Res. | Tokens |
|-------|-----------|----------|---------|----------------|---------------|---------|--------|
| Chain-of-thought prompting | 0% | 55% | 5.88% | ✓ | 0% | 0% | 56,433 |
| RLHF for LLM alignment | 0% | 20% | 0% | ✗ | 0% | — | 65,097 |
| Vision transformers | 0% | 0% | 53.85% | ✗ | 0% | — | 59,909 |
| RAG for knowledge-intensive NLP | 0% | 20% | 45% | ✗ | 0% | 0% | 44,019 |
| Parameter-efficient fine-tuning | 0% | 0% | 38.46% | ✗ | 0% | — | 32,881 |
| Diffusion models | 0% | 8% | 45.45% | ✓ | — | 0% | 57,503 |
| Scaling laws | 0% | 8% | 35% | ✗ | 0% | — | 43,217 |
| In-context learning | 0% | 0% | 64.29% | ✗ | 0% | — | 61,867 |
| Hallucination in LLMs | 0% | 0% | 20.83% | ✗ | 0% | 0% | 50,949 |
| Sparse mixture of experts | 0% | 8% | 5.26% | ✗ | 0% | 0% | 41,770 |

### Почему метрики плохие

| Проблема | Причина | Влияние |
|----------|---------|---------|
| **Низкий coverage** | Semantic Scholar rate limited без API ключа; ArXiv находит свежие, но не классические работы | Пропускаем важные статьи |
| **Hallucination >0%** | Synthesis генерирует paper_id, которые не проходят HTTP-проверку | Ненадёжные цитаты |
| **Contradiction recall 0%** | Synthesis не детектирует противоречия между конкретными парами статей | Пропускаем научные дискуссии |
| **OQ resolution 0%** | Вторая итерация не закрывает открытые вопросы | Reflection loop неэффективен |

---

## Проблемы агентов

### 1. Hallucination в Synthesis

**Как считается:** Verifier делает HTTP GET на `arxiv.org/abs/{id}` — если 200 OK, цитата verified, иначе removed.

**Причины галлюцинаций:**
- LLM генерирует paper_id "по памяти", а не из найденных статей
- Неправильный формат arxiv ID (например, `2301.1234` вместо `2301.01234`)
- Ссылка на статью, которая была в training data, но не в текущем контексте

**Как улучшить:**
- Строгий промпт: "Используй ТОЛЬКО paper_id из предоставленного списка"
- Structured output с enum возможных paper_id
- Валидация формата arxiv ID перед генерацией

### 2. Низкий Contradiction Recall

**Причина:** Synthesis видит абстракты, но не полные тексты. Противоречия часто в деталях методологии.

**Как улучшить:**
- Явный промпт на поиск противоречий: "Найди статьи с противоположными выводами"
- Few-shot примеры известных научных дискуссий
- Chain-of-thought для сравнения пар статей

### 3. Неэффективный Reflection Loop

**Причина:** Open questions из первой итерации слишком абстрактные, Planner генерирует похожие подзапросы.

**Как улучшить:**
- Более конкретные open questions с указанием, что именно искать
- Запрет на повторение подзапросов из предыдущей итерации
- Анализ gap между найденным и ожидаемым

---

## Методы улучшения промптов

### Применимые техники

| Техника | Описание | Применимость для LitRadar |
|---------|----------|---------------------------|
| **Few-shot prompting** | Примеры входа-выхода в промпте | ✅ Для Critic (примеры релевантных/нерелевантных статей) |
| **Chain-of-Thought (CoT)** | Пошаговое рассуждение | ✅ Для Synthesis (анализ противоречий) |
| **Self-Consistency** | Несколько CoT → majority vote | ⚠️ Дорого, но полезно для Critic |
| **Structured Output** | JSON schema для ответа | ✅ Уже используется, можно усилить |
| **Role prompting** | "Ты эксперт в ML..." | ✅ Для Synthesis и Planner |
| **Constrained generation** | Ограничение vocabulary | ✅ Для paper_id (только из списка) |
| **Self-reflection** | Модель проверяет свой ответ | ⚠️ Увеличивает latency, но снижает hallucination |
| **Retrieval-Augmented Generation** | Контекст из базы знаний | ✅ Уже используется (ChromaDB) |


---

## Структура проекта

```
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py          # Декомпозиция темы на подзапросы
│   │   ├── critic.py           # Фильтрация релевантных статей
│   │   ├── synthesis.py        # Анализ, выявление противоречий
│   │   └── renderer.py         # Формирование финального отчёта
│   ├── search/
│   │   ├── __init__.py
│   │   ├── arxiv_client.py     # ArXiv API (свежие препринты)
│   │   ├── semantic_scholar_client.py  # S2 API (высокоцитируемые)
│   │   ├── openalex_client.py  # OpenAlex API (fallback)
│   │   └── sanitizer.py        # Очистка входных данных
│   ├── orchestrator.py         # LangGraph pipeline
│   ├── retriever.py            # ChromaDB vector store
│   ├── verifier.py             # HTTP verification of citations
│   ├── state.py                # AgentState, PaperMetadata
│   ├── llm_client.py           # OpenRouter LLM client
│   ├── config.py               # Settings from .env
│   └── logger.py               # Langfuse logging
│
├── evals/
│   ├── eval_runner.py          # Evaluation script с preload
│   ├── expand_expected_papers.py  # Расширение эталонных наборов
│   ├── test_topics.json        # 10 тестовых тем с ground truth
│   └── results/                # Результаты (gitignored)
│
├── docs/
│   ├── system-design.md        # Архитектура системы
│   ├── product-proposal.md     # Продуктовое описание
│   ├── governance.md           # Правила разработки
│   ├── prompts/
│   │   ├── prompts-planner.md
│   │   ├── prompts-critic.md
│   │   ├── prompts-synthesis.md
│   │   └── prompts-renderer.md
│   └── specs/
│       ├── orchestrator.md
│       ├── retriever.md
│       ├── memory.md
│       ├── tools.md
│       ├── observability.md
│       └── serving.md
│
├── app.py                      # Streamlit UI
├── pyproject.toml              # Зависимости (uv)
├── uv.lock                     # Lock-файл
├── .env.example                # Шаблон переменных окружения
├── .gitignore
└── README.md
```

---

## Лицензия

MIT
