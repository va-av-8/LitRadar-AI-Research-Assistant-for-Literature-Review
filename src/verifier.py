"""HTTP verification of paper citations."""

import asyncio
import httpx
from typing import List, Tuple

from .state import AgentState, Citation
from .config import get_settings
from .logger import logger

# Longer timeout for slow ArXiv responses
VERIFICATION_TIMEOUT = 15


async def verify_arxiv(paper_id: str, client: httpx.AsyncClient) -> Tuple[str, bool]:
    """Verify an ArXiv paper exists via HTTP."""
    arxiv_id = paper_id.replace("arxiv:", "")
    url = f"https://export.arxiv.org/abs/{arxiv_id}"

    try:
        response = await client.get(url, follow_redirects=True)
        return (paper_id, response.status_code == 200)
    except Exception:
        return (paper_id, False)


async def verify_semantic_scholar(paper_id: str, client: httpx.AsyncClient) -> Tuple[str, bool]:
    """Verify a Semantic Scholar paper exists via HTTP."""
    s2_id = paper_id.replace("s2:", "")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}"

    try:
        response = await client.get(url)
        if response.status_code == 200:
            data = response.json()
            return (paper_id, data.get("paperId") == s2_id)
        return (paper_id, False)
    except Exception:
        return (paper_id, False)


async def verify_papers_batch(paper_ids: List[str]) -> dict:
    """Verify multiple papers concurrently."""
    settings = get_settings()
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    results = {}
    async with httpx.AsyncClient(timeout=VERIFICATION_TIMEOUT, headers=headers) as client:
        tasks = []
        for paper_id in paper_ids:
            if paper_id.startswith("arxiv:"):
                tasks.append(verify_arxiv(paper_id, client))
            elif paper_id.startswith("s2:"):
                tasks.append(verify_semantic_scholar(paper_id, client))

        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for result in completed:
                if isinstance(result, tuple):
                    paper_id, is_valid = result
                    results[paper_id] = is_valid
                # Exceptions are treated as verification failure

    return results


def verifier_node(state: AgentState) -> AgentState:
    """LangGraph node: HTTP verification of citations."""
    if state.get("error"):
        return state

    if not state.get("synthesis_output") or not state["synthesis_output"].get("citations"):
        state["verified_citations"] = []
        return state

    citations = state["synthesis_output"]["citations"]

    # Get unique paper_ids to verify
    paper_ids = list(set(c["paper_id"] for c in citations))

    # Batch verification
    verification_results = asyncio.run(verify_papers_batch(paper_ids))

    verified: List[Citation] = []
    removed: List[Citation] = []

    for citation in citations:
        paper_id = citation["paper_id"]
        is_valid = verification_results.get(paper_id, False)

        if is_valid:
            verified.append(citation)
        else:
            removed.append(citation)
            logger.warning(
                "citation_verification_failed",
                session_id=state["session_id"],
                agent_name="verifier",
                paper_id=paper_id,
                claim=citation["claim"],
            )

    state["verified_citations"] = verified
    state["removed_citations"].extend(removed)

    # Check if all citations failed
    if citations and not verified:
        state["error"] = "All citations failed HTTP verification. Cannot produce a reliable review."
        logger.error(
            "all_citations_failed",
            session_id=state["session_id"],
            agent_name="verifier",
            total_citations=len(citations),
        )

    logger.info(
        "verification_complete",
        session_id=state["session_id"],
        agent_name="verifier",
        verified=len(verified),
        removed=len(removed),
    )

    return state
