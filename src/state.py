"""AgentState TypedDict and related data structures."""

from typing import TypedDict, Optional, List, Set
import uuid


class PaperMetadata(TypedDict):
    """Metadata for a single paper."""
    paper_id: str          # "arxiv:..." or "s2:..."
    title: str
    authors: List[str]
    abstract: Optional[str]
    year: Optional[int]
    source: str            # "arxiv" | "semantic_scholar"
    citation_count: Optional[int]


class Contradiction(TypedDict):
    """A detected contradiction between two papers."""
    topic: str
    paper_a: str           # paper_id
    paper_b: str           # paper_id
    description: str


class Citation(TypedDict):
    """A citation linking a claim to a paper."""
    paper_id: str
    claim: str             # the specific claim that cites the paper


class SynthesisSection(TypedDict):
    """A thematic section in the synthesis."""
    title: str
    summary: str
    paper_ids: List[str]


class SynthesisJSON(TypedDict):
    """Structured output from Synthesis agent."""
    sections: List[SynthesisSection]
    citations: List[Citation]
    detected_contradictions: List[Contradiction]
    open_questions: List[str]


class AgentState(TypedDict):
    """Central state passed between all LangGraph nodes."""

    # Input data
    query: str

    # Iteration control
    iteration: int                          # current iteration, starting from 0
    max_iterations: int                     # = 2 (from config)
    prev_open_questions: List[str]          # for detecting loop stall

    # Planning
    subqueries: List[str]

    # Knowledge base (from persistent ChromaDB, session start)
    papers_from_kb: List[PaperMetadata]

    # Search
    papers_seen: Set[str]                   # all paper_ids in session (deduplication)
    papers_found: List[PaperMetadata]       # papers from current iteration (for Critic)
    papers_accepted: List[PaperMetadata]    # accumulated accepted papers (for Synthesis)
    total_papers_searched: int              # total papers found across all iterations

    # Synthesis
    synthesis_output: Optional[SynthesisJSON]

    # Verification
    verified_citations: List[Citation]
    removed_citations: List[Citation]

    # Final output
    final_review: Optional[str]             # Markdown from Renderer

    # Infrastructure
    session_id: str                         # generated at start: f"sess_{uuid4().hex[:8]}"
    sources_available: List[str]            # ["arxiv", "semantic_scholar"]
    token_count: int                        # total tokens in session
    session_budget_remaining: float         # remaining budget in $
    has_injection_signal: bool
    injection_rate: float                   # fraction of abstracts with injection patterns
    source_gap_note: Optional[str]          # note about missing source
    error: Optional[str]                    # error message for UI


def create_initial_state(query: str, max_iterations: int = 2, budget_hard_limit: float = 0.15) -> AgentState:
    """Create an initial AgentState for a new session."""
    return AgentState(
        query=query,
        iteration=0,
        max_iterations=max_iterations,
        prev_open_questions=[],
        subqueries=[],
        papers_from_kb=[],
        papers_seen=set(),
        papers_found=[],
        papers_accepted=[],
        total_papers_searched=0,
        synthesis_output=None,
        verified_citations=[],
        removed_citations=[],
        final_review=None,
        session_id=f"sess_{uuid.uuid4().hex[:8]}",
        sources_available=["arxiv", "semantic_scholar"],
        token_count=0,
        session_budget_remaining=budget_hard_limit,
        has_injection_signal=False,
        injection_rate=0.0,
        source_gap_note=None,
        error=None,
    )
