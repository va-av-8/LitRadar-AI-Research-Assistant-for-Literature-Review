"""OpenAlex API client - fallback for Semantic Scholar rate limits."""

import time
from typing import List, Set, Dict, Any
import httpx

from ..state import PaperMetadata
from ..config import get_settings
from ..logger import logger


def search_openalex(
    query: str,
    max_results: int = 10,
    papers_seen: Set[str] = None,
    session_id: str = "",
    min_citations: int = 10,
) -> List[PaperMetadata]:
    """
    Search OpenAlex for papers matching query.

    Fallback for Semantic Scholar when rate limited.
    Filters by AI/ML topics for better relevance.

    Args:
        query: Search query
        max_results: Maximum results to return
        papers_seen: Set of paper_ids to exclude (deduplication)
        session_id: For logging
        min_citations: Minimum citation count filter

    Returns:
        List of PaperMetadata
    """
    papers_seen = papers_seen or set()
    settings = get_settings()

    url = "https://api.openalex.org/works"

    # Fetch more results to filter locally
    fetch_count = min(max_results * 5, 100)

    # AI/ML topic IDs for relevance filtering:
    # T10181 = Natural Language Processing Techniques
    # T10320 = Neural Networks and Applications
    # T10036 = Advanced Neural Network Applications
    # T12549 = Image and Object Detection Techniques
    topics_filter = "topics.id:T10181|T10320|T10036|T12549"

    params = {
        "search": query,
        "filter": f"{topics_filter},indexed_in:arxiv,cited_by_count:>{min_citations},publication_year:>2018",
        "per_page": fetch_count,
        "select": "id,title,authorships,publication_year,cited_by_count,abstract_inverted_index,locations",
    }

    try:
        with httpx.Client(timeout=settings.api_timeout_seconds) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        papers: List[PaperMetadata] = []

        for work in data.get("results", []):
            # Extract arxiv ID from locations
            arxiv_id = None
            for location in work.get("locations", []):
                landing_url = location.get("landing_page_url", "")
                if "arxiv.org/abs/" in landing_url:
                    arxiv_id = landing_url.split("arxiv.org/abs/")[-1]
                    break

            if not arxiv_id:
                continue

            paper_id = f"arxiv:{arxiv_id}"

            # Skip if already seen
            if paper_id in papers_seen:
                continue

            # Reconstruct abstract from inverted index
            abstract = None
            inverted = work.get("abstract_inverted_index")
            if inverted:
                try:
                    words = [""] * (max(max(pos) for pos in inverted.values()) + 1)
                    for word, positions in inverted.items():
                        for pos in positions:
                            words[pos] = word
                    abstract = " ".join(words)
                except (ValueError, TypeError):
                    abstract = None

            # Extract authors
            authors = []
            for authorship in work.get("authorships", [])[:10]:
                author = authorship.get("author", {})
                name = author.get("display_name")
                if name:
                    authors.append(name)

            papers.append(PaperMetadata(
                paper_id=paper_id,
                title=work.get("title", ""),
                authors=authors,
                abstract=abstract,
                year=work.get("publication_year"),
                source="openalex",
                citation_count=work.get("cited_by_count"),
            ))

        # Sort by citation count and return top N
        papers.sort(key=lambda x: x.citation_count or 0, reverse=True)
        papers = papers[:max_results]

        logger.info(
            "openalex_search",
            session_id=session_id,
            agent_name="search",
            query=query,
            results=len(papers),
        )

        # Small delay to be nice to the API
        time.sleep(0.2)

        return papers

    except Exception as e:
        logger.error(
            "openalex_search_failed",
            session_id=session_id,
            agent_name="search",
            query=query,
            error=str(e),
        )
        return []
