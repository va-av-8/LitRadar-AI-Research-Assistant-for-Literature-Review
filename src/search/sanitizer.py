"""Sanitization layer for injection pattern detection and removal."""

import re
from typing import Tuple, List

from ..state import AgentState, PaperMetadata
from ..logger import logger


# Injection patterns to detect and remove
INJECTION_PATTERNS = [
    "ignore",
    "disregard",
    "you are now",
    "new instructions",
    "system:",
    "[INST]",
    "forget previous",
    "override",
]


def sanitize(abstract: str) -> Tuple[str, bool]:
    """
    Sanitize abstract by removing injection patterns.

    Returns:
        Tuple of (cleaned_text, was_flagged)
    """
    if not abstract:
        return abstract, False

    # Check for patterns (case-insensitive)
    abstract_lower = abstract.lower()
    flagged = any(pattern.lower() in abstract_lower for pattern in INJECTION_PATTERNS)

    # Remove patterns if found
    if flagged:
        pattern_regex = "|".join(re.escape(p) for p in INJECTION_PATTERNS)
        cleaned = re.sub(pattern_regex, "[REMOVED]", abstract, flags=re.IGNORECASE)
        return cleaned, True

    return abstract, False


def wrap_in_external_content(abstract: str, source: str, paper_id: str) -> str:
    """Wrap abstract in external_content tag."""
    return f'<external_content source="{source}" paper_id="{paper_id}">\n{abstract}\n</external_content>'


def sanitization_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Sanitize papers_found and detect injection attempts.

    - Cleans injection patterns from abstracts
    - Wraps abstracts in <external_content> tags
    - Updates injection_rate and has_injection_signal in state
    """
    if state.get("error"):
        return state

    papers_found = state.get("papers_found", [])
    if not papers_found:
        return state

    flagged_count = 0
    sanitized_papers: List[PaperMetadata] = []

    for paper in papers_found:
        abstract = paper.get("abstract") or ""

        # Sanitize
        cleaned_abstract, flagged = sanitize(abstract)

        if flagged:
            flagged_count += 1
            logger.security_warning(
                "injection_signal",
                session_id=state["session_id"],
                agent_name="sanitizer",
                paper_id=paper["paper_id"],
            )

        # Wrap in external_content tag
        if cleaned_abstract:
            wrapped = wrap_in_external_content(
                cleaned_abstract, paper["source"], paper["paper_id"]
            )
        else:
            # No abstract available
            fallback_text = f"[Abstract not available. Title: {paper['title']}. Authors: {', '.join(paper.get('authors', []))}.]"
            wrapped = wrap_in_external_content(
                fallback_text, paper["source"], paper["paper_id"]
            )

        # Create new paper with sanitized abstract
        sanitized_paper = PaperMetadata(
            paper_id=paper["paper_id"],
            title=paper["title"],
            authors=paper.get("authors", []),
            abstract=wrapped,
            year=paper.get("year"),
            source=paper["source"],
            citation_count=paper.get("citation_count"),
        )
        sanitized_papers.append(sanitized_paper)

    # Update state
    state["papers_found"] = sanitized_papers
    state["has_injection_signal"] = flagged_count > 0
    state["injection_rate"] = flagged_count / len(papers_found) if papers_found else 0.0

    logger.info(
        "sanitization_complete",
        session_id=state["session_id"],
        agent_name="sanitizer",
        total_papers=len(papers_found),
        flagged=flagged_count,
        injection_rate=state["injection_rate"],
    )

    return state
