"""Planner agent for decomposing research topics into subqueries."""

from typing import List

from ..state import AgentState
from ..llm_client import get_llm_client
from ..logger import logger

# System prompt - exactly as specified in prompts-planner.md
PLANNER_SYSTEM_PROMPT = """You are a search query planner for an academic literature review system focused on AI/ML research.

Your task is to decompose a research topic into 4–6 specific search subqueries suitable for querying ArXiv and Semantic Scholar.

## Rules

1. Each subquery must be specific enough to return focused academic results — think of it as a query a researcher would type into Google Scholar.
2. Structure subqueries to cover these four categories — aim to have at least one subquery per category:
   - **Methods**: how X works, core mechanisms, architectures, algorithms
   - **Benchmarks**: empirical evaluation, performance comparison, datasets
   - **Limitations**: failure cases, when X does not work, constraints, challenges
   - **Criticism**: opposing views, skeptical perspectives, "does X actually work", X overestimated, mirage, alternative explanations
   If a topic genuinely lacks material for one category (e.g., no established benchmarks), skip it — do not invent subqueries.
3. Do not generate more than 6 subqueries. Minimum 4.
4. Subqueries must be in English, regardless of the language of the original query.
5. Do not include author names, venue names, or years in subqueries.
6. The Criticism category is mandatory — always include at least one subquery targeting critical or opposing perspectives. This is essential for detecting contradictions in the literature.
   Important: criticism subqueries do not need to contain the topic's exact terminology. Opposing work often uses different terms (e.g., a critique of chain-of-thought prompting may be framed as a paper about "emergent abilities mirage" or "scaling laws" without mentioning CoT directly). Think about what the counter-position would be called in the literature, not just what critics say about the topic by name.

## Output format

Respond with a JSON object only. No preamble, no explanation outside the JSON.

{
  "subqueries": ["...", "...", "..."],
  "reasoning": "one sentence explaining the decomposition logic"
}

## Good decomposition — example

Topic: "chain-of-thought prompting in large language models"

Good:
{
  "subqueries": [
    "chain-of-thought prompting mechanisms reasoning large language models",
    "chain-of-thought few-shot zero-shot prompting benchmark comparison",
    "chain-of-thought prompting small models failure cases limitations",
    "emergent abilities large language models mirage artifact metric choice"
  ],
  "reasoning": "Covers methods (core CoT mechanism), benchmarks (few-shot vs zero-shot comparison), limitations (small model failure), and criticism (emergent abilities debate — using opposing literature's own terminology rather than 'criticism of CoT')."
}

Bad (do not do this):
{
  "subqueries": [
    "chain-of-thought prompting",
    "CoT prompting LLMs",
    "chain of thought in language models",
    "CoT reasoning transformers",
    "chain-of-thought GPT"
  ],
  "reasoning": "..."
}
Why bad: all five subqueries retrieve the same papers; no facet diversity.

## Bad decomposition — too broad

Topic: "machine learning"
Do not attempt to decompose this. Return:
{
  "subqueries": [],
  "reasoning": "Topic is too broad to decompose into meaningful academic subqueries."
}

Note: in practice, overly broad topics are blocked by Input Guard before reaching you. This fallback is for edge cases."""


def _build_user_prompt_iter1(query: str, papers_from_kb: List[dict]) -> str:
    """Build user prompt for iteration 1."""
    if papers_from_kb:
        titles = "\n".join(f"- {p['title']}" for p in papers_from_kb[:10])
        return f"""Research topic: {query}

Papers already in the knowledge base (do not generate subqueries that would primarily retrieve these — focus on gaps):
{titles}

Generate 4–6 search subqueries to find papers not yet covered by the knowledge base. Include at least one subquery targeting critical or opposing perspectives."""
    else:
        return f"""Research topic: {query}

The knowledge base is empty. Generate 4–6 search subqueries to broadly cover the key facets of this topic. Include at least one subquery targeting critical or opposing perspectives."""


def _build_user_prompt_iter2(
    query: str,
    open_questions: List[str],
    prev_subqueries: List[str],
) -> str:
    """Build user prompt for iteration 2."""
    oq_text = "\n".join(f"- {q}" for q in open_questions)
    prev_sq_text = "\n".join(f"- {sq}" for sq in prev_subqueries)

    return f"""Research topic: {query}

The first round of search and synthesis identified the following open questions that were not resolved by the papers found so far:

{oq_text}

Generate 4–6 search subqueries specifically targeting these open questions. Do not repeat subqueries from the previous iteration. Include at least one subquery targeting critical or opposing perspectives if not already covered.

Previous subqueries (for reference, do not reuse):
{prev_sq_text}"""


def planner_node(state: AgentState) -> AgentState:
    """LangGraph node: Planner agent."""
    if state.get("error"):
        return state

    client = get_llm_client()
    iteration = state["iteration"]

    # Build appropriate user prompt
    if iteration == 0:
        user_prompt = _build_user_prompt_iter1(
            state["query"],
            state.get("papers_from_kb", []),
        )
    else:
        user_prompt = _build_user_prompt_iter2(
            state["query"],
            state.get("prev_open_questions", []),
            state.get("subqueries", []),
        )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result, input_tokens, output_tokens = client.chat_completion_json(
            messages=messages,
            temperature=0.3,
            max_tokens=300,
            session_id=state["session_id"],
            agent_name="planner",
            retries=2,
        )

        # Update token count
        state["token_count"] += input_tokens + output_tokens

        # Extract subqueries
        subqueries = result.get("subqueries", [])

        # Validate and adjust
        if len(subqueries) > 6:
            logger.warning(
                "planner_too_many_subqueries",
                session_id=state["session_id"],
                agent_name="planner",
                count=len(subqueries),
            )
            subqueries = subqueries[:6]

        if len(subqueries) < 3:
            logger.warning(
                "planner_few_subqueries",
                session_id=state["session_id"],
                agent_name="planner",
                count=len(subqueries),
            )

        if not subqueries:
            # Too broad fallback
            state["error"] = "Topic is too broad to decompose. Please provide a more specific research topic."
            return state

        state["subqueries"] = subqueries

        logger.info(
            "planner_complete",
            session_id=state["session_id"],
            agent_name="planner",
            iteration=iteration,
            subqueries=subqueries,
            reasoning=result.get("reasoning", ""),
        )

    except Exception as e:
        logger.error(
            "planner_failed",
            session_id=state["session_id"],
            agent_name="planner",
            error=str(e),
        )
        state["error"] = f"Planner failed: {str(e)}"

    return state
