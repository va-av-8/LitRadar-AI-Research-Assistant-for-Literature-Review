"""ArXiv API client."""

import time
from typing import List, Set
import arxiv

from ..state import PaperMetadata
from ..config import get_settings
from ..logger import logger


def search_arxiv(
    query: str,
    max_results: int = 10,
    papers_seen: Set[str] = None,
    session_id: str = "",
) -> List[PaperMetadata]:
    """
    Search ArXiv for papers matching query.

    Args:
        query: Search query
        max_results: Maximum results to return
        papers_seen: Set of paper_ids to exclude (deduplication)
        session_id: For logging

    Returns:
        List of PaperMetadata
    """
    papers_seen = papers_seen or set()
    settings = get_settings()

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: List[PaperMetadata] = []
        client = arxiv.Client()

        # Collect all results first
        results = list(client.results(search))

        for result in results:
            # Extract arxiv ID from entry_id URL
            # Format: http://arxiv.org/abs/2301.07597v1 -> arxiv:2301.07597
            entry_id = result.entry_id.split("/")[-1]
            # Remove version suffix if present
            if "v" in entry_id:
                entry_id = entry_id.rsplit("v", 1)[0]
            paper_id = f"arxiv:{entry_id}"

            # Skip if already seen
            if paper_id in papers_seen:
                continue

            papers.append(PaperMetadata(
                paper_id=paper_id,
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                year=result.published.year if result.published else None,
                source="arxiv",
                citation_count=None,  # ArXiv doesn't provide this
            ))

        logger.info(
            "arxiv_search",
            session_id=session_id,
            agent_name="search",
            query=query,
            results=len(papers),
            raw_results=len(results),
            papers_seen_count=len(papers_seen),
        )

        # Respect rate limit between queries (not between results)
        time.sleep(0.5)

        return papers

    except Exception as e:
        logger.error(
            "arxiv_search_failed",
            session_id=session_id,
            agent_name="search",
            query=query,
            error=str(e),
        )
        return []
