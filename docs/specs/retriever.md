# Spec: Retriever

## Роль

Retriever — внешняя база знаний системы и RAG-контур для Synthesis. Решает три задачи:
- Обогащение контекста Planner'а знаниями из предыдущих сессий
- Индексирование новых статей, прошедших фильтрацию Critic'а
- Семантический retrieval для Synthesis по накопленному корпусу

---

## Индекс

| Параметр | Значение |
|---|---|
| Хранилище | ChromaDB, персистентная коллекция на диске (`./chroma_db/papers`) |
| Инициализация | `chromadb.PersistentClient(path="./chroma_db")` |
| Embedding model | `BAAI/bge-small-en-v1.5` (sentence-transformers, 768 dim, локально) |
| Метаданные документа | `paper_id`, `title`, `authors`, `abstract`, `year`, `source`, `citation_count` — полное соответствие `PaperMetadata` из `state.py` |
| Дедупликация | По `paper_id` до индексирования — дубликаты не добавляются |
| Источник текста | Абстракт; если отсутствует (Semantic Scholar) — `title + authors` |

---

## Режимы работы

### 1. KB lookup (начало сессии)

Вызывается сразу после Input Guard, до Planner.

```python
def kb_lookup(query: str, top_k: int = 10) -> List[PaperMetadata]
```

- Делает retrieval по исходному запросу пользователя
- Возвращает `papers_from_kb` → записывается в AgentState
- Planner получает заголовки этих статей и учитывает при декомпозиции запроса
- Если коллекция пуста (первая сессия) — возвращает пустой список, пайплайн продолжается

### 2. Индексирование (после Critic)

```python
def index(papers: List[PaperMetadata]) -> int  # возвращает кол-во добавленных
```

- Принимает `papers_accepted` текущей сессии
- Пропускает `paper_id`, уже присутствующие в коллекции
- Логирует количество добавленных / пропущенных документов

### 3. RAG retrieval (Synthesis)

```python
def retrieve(query: str, top_k: int = 5) -> List[PaperChunk]
```

- Вызывается Synthesis для каждого тематического кластера
- Поиск по всему корпусу (KB + индексированные в текущей сессии)
- Вычисление схожести: cosine similarity по векторам — детерминированное, без LLM-вызовов
- Эмбеддинги генерируются локально через `sentence-transformers` (`BAAI/bge-small-en-v1.5`) — при индексировании и при каждом retrieval-запросе; внешние API не используются

---

## Параметры top-k

| Режим | top_k | Вызывающий |
|---|---|---|
| KB lookup | 10 | Orchestrator (до Planner) |
| RAG retrieval per cluster | 5 | Synthesis Agent |

---

## Reranking

В PoC reranking не используется — cosine similarity достаточно для задачи. В будущих итерациях: cross-encoder reranking (`ms-marco-MiniLM`).

---

## Ограничения

- Индекс не версионируется; при необходимости сбросить — удалить `./chroma_db/`
- Не поддерживает конкурентную запись из нескольких процессов (PoC — одиночный Streamlit)
- Абстракты без полного текста: семантическое покрытие статьи ограничено
- Фильтр по году (`year >= 2018`) опционален, настраивается через config
