"""Critic agent for evaluating paper relevance."""

from typing import List, Dict, Any

from ..state import AgentState, PaperMetadata
from ..llm_client import get_llm_client
from ..config import get_settings
from ..logger import logger


def normalize_paper_id(paper_id: str) -> str:
    """Normalize paper_id by adding prefix if missing.

    LLM often returns '2201.11903' instead of 'arxiv:2201.11903'.
    We detect S2 IDs by their format (40-char hex hash) and treat
    everything else as ArXiv.
    """
    if not paper_id:
        return paper_id
    paper_id = paper_id.strip()
    # Already has prefix
    if paper_id.startswith(("arxiv:", "s2:")):
        return paper_id
    # S2 IDs are 40-character hex hashes (SHA-1)
    if len(paper_id) == 40 and all(c in "0123456789abcdefABCDEF" for c in paper_id):
        return f"s2:{paper_id}"
    # Everything else is ArXiv (new format 2201.11903, old format hep-ph/9901234, etc.)
    return f"arxiv:{paper_id}"

# System prompt - exactly as specified in prompts-critic.md
CRITIC_SYSTEM_PROMPT = """You are a relevance filter for an academic literature review system focused on AI/ML research.

You will receive a research topic and a batch of papers (title + abstract). Your task is to decide whether each paper is relevant to the research topic and should be included in the literature review.

## Relevance criterion

A paper is relevant if it directly addresses the research topic — its core subject matter, methods, findings, or limitations.

A paper is NOT relevant if:
- It merely cites or mentions the topic in passing
- It addresses a related but different problem (e.g., a paper about retrieval augmentation is not relevant to a topic about chain-of-thought prompting, even if both involve LLMs)
- It is a general survey that happens to touch on the topic among many others

A paper IS relevant even if it does not directly mention the topic by name, if it:
- Critically challenges or refutes a foundational concept that the topic depends on (e.g., a paper arguing that emergent abilities in LLMs are a measurement artifact is relevant to chain-of-thought prompting, because CoT's effectiveness is often explained through emergence)
- Provides evidence that substantially changes how the topic should be interpreted or scoped

Judge relevance against the ORIGINAL research topic, not the subquery that retrieved the paper.

## Confidence levels

- high: you are confident in the decision
- medium: the paper is on the boundary; include it if in doubt (err toward inclusion)
- low: the abstract is too vague to judge — mark as relevant and flag for human review

## Output format

Respond with a JSON array only. One object per paper. No preamble, no explanation outside the JSON.

[
  {
    "paper_id": "...",
    "relevant": true,
    "reason": "one sentence explaining the decision",
    "confidence": "high" | "medium" | "low"
  }
]

## Examples

Research topic: "chain-of-thought prompting in large language models"

Paper A — Title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
Abstract: "We explore how generating a chain of thought — a series of intermediate reasoning steps — significantly improves the ability of large language models to perform complex reasoning..."
→ {"paper_id": "arxiv:2201.11903", "relevant": true, "reason": "Directly introduces and evaluates chain-of-thought prompting in LLMs.", "confidence": "high"}

Paper B — Title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
Abstract: "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG), combining parametric and non-parametric memory for language generation..."
→ {"paper_id": "arxiv:2005.11401", "relevant": false, "reason": "Addresses retrieval-augmented generation, not chain-of-thought prompting.", "confidence": "high"}

Paper C — Title: "Emergent Abilities of Large Language Models"
Abstract: "We investigate emergent abilities — capabilities not present in smaller models but appearing in larger ones. We survey tasks including arithmetic, multi-step reasoning, and chain-of-thought prompting..."
→ {"paper_id": "arxiv:2206.07682", "relevant": true, "reason": "Directly analyses chain-of-thought as an emergent ability in the context of LLM scaling.", "confidence": "medium"}

Paper D — Title: "A Survey of Large Language Models"
Abstract: "This survey reviews the development of large language models, covering pre-training, instruction tuning, alignment, and applications including reasoning and prompting techniques..."
→ {"paper_id": "arxiv:2303.18223", "relevant": false, "reason": "Broad survey of LLMs; chain-of-thought is one of many topics covered, not the focus.", "confidence": "medium"}"""


def _format_papers_batch(papers: List[PaperMetadata]) -> str:
    """Format papers for critic prompt with external_content tags."""
    parts = []
    for paper in papers:
        paper_id = paper['paper_id']
        source = paper.get('source', 'arxiv')
        title = paper['title']
        abstract = paper.get("abstract", "")
        authors = "; ".join(paper.get("authors", [])[:3])
        if len(paper.get("authors", [])) > 3:
            authors += "; et al."

        # Format content based on abstract availability
        if abstract:
            content = abstract
        else:
            content = f"[Abstract not available. Title: {title}. Authors: {authors}.]"

        parts.append(f"""---
paper_id: {paper_id}
title: {title}
<external_content source="{source}" paper_id="{paper_id}">
{content}
</external_content>""")
    return "\n\n".join(parts)


def critic_node(state: AgentState) -> AgentState:
    """LangGraph node: Critic agent evaluates paper relevance in batches."""
    if state.get("error"):
        return state

    papers_found = state.get("papers_found", [])
    if not papers_found:
        logger.warning(
            "critic_no_papers",
            session_id=state["session_id"],
            agent_name="critic",
        )
        return state

    client = get_llm_client()
    settings = get_settings()
    batch_size = settings.critic_batch_size

    all_accepted: List[PaperMetadata] = []
    all_rejected_count = 0

    # Process in batches
    for i in range(0, len(papers_found), batch_size):
        batch = papers_found[i:i + batch_size]
        papers_batch_text = _format_papers_batch(batch)

        user_prompt = f"""Research topic: {state['query']}

Evaluate the relevance of the following papers:

{papers_batch_text}"""

        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result, input_tokens, output_tokens = client.chat_completion_json(
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
                session_id=state["session_id"],
                agent_name="critic",
                retries=2,
            )

            state["token_count"] += input_tokens + output_tokens

            # Handle list, or object with "decisions" or "papers" key
            if isinstance(result, list):
                decisions = result
            elif isinstance(result, dict):
                decisions = result.get("decisions") or result.get("papers") or []
            else:
                decisions = []

            # Create lookup by paper_id with normalization
            decisions_map: Dict[str, Dict[str, Any]] = {}
            for d in decisions:
                if isinstance(d, dict) and "paper_id" in d:
                    normalized_id = normalize_paper_id(d["paper_id"])
                    decisions_map[normalized_id] = d

            # Process each paper in batch
            for paper in batch:
                decision = decisions_map.get(paper["paper_id"])

                if decision is None:
                    # Missing decision - default to relevant
                    logger.warning(
                        "critic_missing_decision",
                        session_id=state["session_id"],
                        agent_name="critic",
                        paper_id=paper["paper_id"],
                    )
                    all_accepted.append(paper)
                elif decision.get("relevant", True):
                    all_accepted.append(paper)
                    logger.info(
                        "paper_accepted",
                        session_id=state["session_id"],
                        agent_name="critic",
                        paper_id=paper["paper_id"],
                        reason=decision.get("reason", ""),
                        confidence=decision.get("confidence", ""),
                    )
                else:
                    all_rejected_count += 1
                    logger.info(
                        "paper_rejected",
                        session_id=state["session_id"],
                        agent_name="critic",
                        paper_id=paper["paper_id"],
                        reason=decision.get("reason", ""),
                    )

        except Exception as e:
            # Fallback: accept all papers in batch
            logger.error(
                "critic_batch_failed",
                session_id=state["session_id"],
                agent_name="critic",
                batch_start=i,
                error=str(e),
            )
            all_accepted.extend(batch)

    # Extend papers_accepted (accumulate across iterations)
    state["papers_accepted"].extend(all_accepted)

    # Add to papers_seen
    for paper in all_accepted:
        state["papers_seen"].add(paper["paper_id"])

    logger.info(
        "critic_complete",
        session_id=state["session_id"],
        agent_name="critic",
        accepted=len(all_accepted),
        rejected=all_rejected_count,
        total=len(papers_found),
    )

    return state
