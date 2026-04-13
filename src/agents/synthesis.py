"""Synthesis agent for grouping papers and detecting contradictions."""

from typing import List

from ..state import AgentState, PaperMetadata, SynthesisJSON
from ..llm_client import get_llm_client
from ..retriever import get_retriever
from ..config import get_settings
from ..logger import logger

# System prompt - exactly as specified in prompts-synthesis.md
SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis agent for an academic literature review system.

You will receive a research topic, metadata of accepted papers, and relevant abstract excerpts retrieved from a knowledge base. Your task is to produce a structured literature review in JSON format.

## Your tasks

1. Group papers into thematic sections (2–5 sections). Each section covers a coherent sub-topic.
2. Write a concise summary for each section (3–6 sentences) based on the abstracts provided.
3. For each factual claim in the summaries, add a citation pointing to the paper that supports it.
4. Identify genuine contradictions between papers — cases where two papers make directly opposing claims about the same specific finding.
5. Formulate open questions that the current set of papers does not answer.

## Citation rules — CRITICAL

- Only cite papers from the provided list (papers_accepted + papers_from_kb).
- Every paper_id in citations must exactly match a paper_id from the provided list.
- Do not invent, guess, or abbreviate paper_ids. Copy them verbatim.
- If you cannot support a claim with a paper from the list, do not make the claim.

## Contradiction rules

A contradiction is valid ONLY if:
- Two papers make opposing claims about the SAME specific phenomenon (e.g., "CoT improves accuracy on tasks below 7B parameters" vs. "CoT degrades performance on models below 7B parameters")
- The opposition is explicit, not implied by different methodologies or different datasets

Do NOT flag as contradiction:
- Papers studying different model sizes and reaching different conclusions (different scope, not contradiction)
- Papers using different metrics that are not directly comparable
- A newer paper improving on an older one (that is progress, not contradiction)

## Open questions rules

An open question is something that:
- Is directly relevant to the research topic
- Is raised by contradictions or gaps in the current paper set
- Cannot be answered from the abstracts provided

Do not manufacture open questions. If everything is well-covered, return an empty list.

## Output format

Respond with a JSON object only. No preamble, no markdown, no explanation outside the JSON.

{
  "sections": [
    {
      "title": "short descriptive title",
      "summary": "3–6 sentence summary of papers in this section",
      "paper_ids": ["paper_id_1", "paper_id_2"]
    }
  ],
  "citations": [
    {
      "paper_id": "exact paper_id from the provided list",
      "claim": "the specific claim in the summary that this paper supports"
    }
  ],
  "detected_contradictions": [
    {
      "topic": "the specific phenomenon in dispute",
      "paper_a": "paper_id",
      "paper_b": "paper_id",
      "description": "Paper A claims X, Paper B claims Y, both about Z"
    }
  ],
  "open_questions": [
    "Specific question that the current papers do not answer"
  ]
}

## Example — valid contradiction

Topic: "chain-of-thought prompting in large language models"

paper_a: arxiv:2201.11903 — "CoT prompting consistently degrades performance in models with fewer than 7B parameters"
paper_b: arxiv:2309.05653 — "Few-shot CoT improves accuracy on arithmetic tasks even in 3B parameter models when examples are carefully selected"

Valid contradiction:
{
  "topic": "CoT effectiveness in sub-7B parameter models",
  "paper_a": "arxiv:2201.11903",
  "paper_b": "arxiv:2309.05653",
  "description": "Paper A reports consistent degradation from CoT in models below 7B parameters; Paper B shows accuracy gains with few-shot CoT in a 3B model, suggesting the relationship depends on example selection."
}

## Example — invalid contradiction (do not flag)

paper_a evaluated GPT-3 on arithmetic benchmarks; paper_b evaluated LLaMA-2 on commonsense reasoning.
→ Different models, different tasks. Not a contradiction — different scope."""


def _format_papers_metadata(papers: List[PaperMetadata]) -> str:
    """Format papers metadata for synthesis prompt."""
    parts = []
    for p in papers:
        authors = "; ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += "; et al."
        citation_count = p.get("citation_count")
        cc_str = f"  citation_count: {citation_count}" if citation_count else ""

        parts.append(f"""- paper_id: {p['paper_id']}
  title: {p['title']}
  authors: {authors}
  year: {p.get('year', 'N/A')}{cc_str}""")
    return "\n".join(parts)


def _format_rag_abstracts(subqueries: List[str], session_id: str) -> str:
    """Retrieve and format abstracts for each subquery.

    Each abstract is prefixed with paper_id to enable proper citation.
    """
    retriever = get_retriever()
    settings = get_settings()

    parts = []
    seen_paper_ids = set()  # Deduplicate across subqueries

    for subquery in subqueries:
        papers = retriever.retrieve(subquery, top_k=settings.retriever_top_k_synthesis)
        if papers:
            abstracts = []
            for p in papers:
                paper_id = p.get("paper_id", "")
                abstract = p.get("abstract", "")
                # Skip duplicates and empty abstracts
                if abstract and paper_id and paper_id not in seen_paper_ids:
                    seen_paper_ids.add(paper_id)
                    # Format: paper_id on its own line, then abstract
                    abstracts.append(f"[{paper_id}]\n{abstract}")
            if abstracts:
                parts.append(f"### Subquery: {subquery}\n\n" + "\n\n".join(abstracts))

    return "\n\n".join(parts) if parts else "[No relevant abstracts retrieved]"


def synthesis_node(state: AgentState) -> AgentState:
    """LangGraph node: Synthesis agent."""
    if state.get("error"):
        return state

    papers_accepted = state.get("papers_accepted", [])
    papers_from_kb = state.get("papers_from_kb", [])

    if not papers_accepted and not papers_from_kb:
        state["error"] = "No papers available for synthesis."
        return state

    client = get_llm_client()
    settings = get_settings()

    # Format metadata
    all_papers = papers_accepted + papers_from_kb
    papers_metadata = _format_papers_metadata(all_papers)

    # RAG retrieval using subqueries as cluster proxies
    rag_abstracts = _format_rag_abstracts(state.get("subqueries", []), state["session_id"])

    user_prompt = f"""Research topic: {state['query']}

## Accepted papers (metadata)

{papers_metadata}

## Relevant abstracts from knowledge base

The following abstracts were retrieved as most relevant to each search subquery. Use them as the primary source for summaries and citations.

{rag_abstracts}

## Instructions

- Base all claims and citations strictly on the abstracts provided above.
- Do not use knowledge from your training data to make claims — only what is in the abstracts.
- If abstracts are insufficient to write a substantive summary for a section, write what you can and note the limitation in the summary text itself."""

    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result, input_tokens, output_tokens = client.chat_completion_json(
            messages=messages,
            model=settings.llm_model_synthesis,
            temperature=0.4,
            max_tokens=4000,
            session_id=state["session_id"],
            agent_name="synthesis",
            retries=3,
        )

        state["token_count"] += input_tokens + output_tokens

        # Validate and construct SynthesisJSON
        synthesis_output = SynthesisJSON(
            sections=result.get("sections", []),
            citations=result.get("citations", []),
            detected_contradictions=result.get("detected_contradictions", []),
            open_questions=result.get("open_questions", []),
        )

        state["synthesis_output"] = synthesis_output

        logger.info(
            "synthesis_complete",
            session_id=state["session_id"],
            agent_name="synthesis",
            sections=len(synthesis_output["sections"]),
            citations=len(synthesis_output["citations"]),
            contradictions=len(synthesis_output["detected_contradictions"]),
            open_questions=len(synthesis_output["open_questions"]),
        )

    except Exception as e:
        logger.error(
            "synthesis_failed",
            session_id=state["session_id"],
            agent_name="synthesis",
            error=str(e),
        )
        # Fallback: create minimal synthesis from metadata
        state["synthesis_output"] = SynthesisJSON(
            sections=[{
                "title": "Papers Overview",
                "summary": f"Found {len(all_papers)} papers on the topic. Synthesis failed due to an error.",
                "paper_ids": [p["paper_id"] for p in all_papers[:10]],
            }],
            citations=[],
            detected_contradictions=[],
            open_questions=[],
        )

    return state
