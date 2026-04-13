"""Semantic Scholar API client."""

import time
from typing import List, Set, Optional
import httpx

from ..state import PaperMetadata
from ..config import get_settings
from ..logger import logger


def search_semantic_scholar(
    query: str,
    max_results: int = 10,
    papers_seen: Set[str] = None,
    session_id: str = "",
    sort_by_citations: bool = False,
) -> List[PaperMetadata]:
    """
    Search Semantic Scholar for papers matching query.

    Args:
        query: Search query
        max_results: Maximum results to return
        papers_seen: Set of paper_ids to exclude (deduplication)
        session_id: For logging
        sort_by_citations: If True, sort results by citation count descending

    Returns:
        List of PaperMetadata
    """
    papers_seen = papers_seen or set()
    settings = get_settings()

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "paperId,externalIds,title,authors,abstract,year,citationCount",
        "limit": max_results,
    }

    if sort_by_citations:
        params["sort"] = "citationCount:desc"

    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    try:
        with httpx.Client(timeout=settings.api_timeout_seconds) as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

        papers: List[PaperMetadata] = []

        for item in data.get("data", []):
            # Prefer arxiv ID if available, otherwise use S2 paper ID
            external_ids = item.get("externalIds", {})
            arxiv_id = external_ids.get("ArXiv")
            if arxiv_id:
                paper_id = f"arxiv:{arxiv_id}"
            else:
                paper_id = f"s2:{item['paperId']}"

            # Skip if already seen
            if paper_id in papers_seen:
                continue

            # Extract author names
            authors = [a.get("name", "") for a in item.get("authors", [])]

            papers.append(PaperMetadata(
                paper_id=paper_id,
                title=item.get("title", ""),
                authors=authors,
                abstract=item.get("abstract"),  # May be None
                year=item.get("year"),
                source="semantic_scholar",
                citation_count=item.get("citationCount"),
            ))

        logger.info(
            "semantic_scholar_search",
            session_id=session_id,
            agent_name="search",
            query=query,
            results=len(papers),
            sort_by_citations=sort_by_citations,
        )

        # Respect rate limit (5s without API key to avoid 429 errors)
        time.sleep(5)

        return papers

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            # Rate limited - fallback to OpenAlex immediately instead of long waits
            logger.warning(
                "semantic_scholar_rate_limited_fallback",
                session_id=session_id,
                agent_name="search",
                query=query,
                fallback="openalex",
            )
            from .openalex_client import search_openalex
            return search_openalex(
                query=query,
                max_results=max_results,
                papers_seen=papers_seen,
                session_id=session_id,
            )
        else:
            logger.error(
                "semantic_scholar_search_failed",
                session_id=session_id,
                agent_name="search",
                query=query,
                status_code=e.response.status_code,
                error=str(e),
            )
        return []

    except Exception as e:
        logger.error(
            "semantic_scholar_search_failed",
            session_id=session_id,
            agent_name="search",
            query=query,
            error=str(e),
        )
        return []
