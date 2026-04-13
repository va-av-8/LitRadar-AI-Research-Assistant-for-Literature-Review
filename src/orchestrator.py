"""LangGraph orchestrator for LitRadar pipeline."""

from typing import Literal
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
import numpy as np

from .state import AgentState, create_initial_state
from .config import get_settings
from .logger import logger
from .llm_client import get_llm_client
from .retriever import kb_lookup_node, retriever_index_node
from .verifier import verifier_node
from .agents import planner_node, critic_node, synthesis_node, renderer_node
from .search import search_arxiv, search_semantic_scholar, sanitization_node


# ============================================================================
# Input Guard
# ============================================================================

STOP_PHRASES = [
    "how to make", "instructions for", "exploit", "hack into",
    "bypass security", "steal", "illegal", "malware",
]

TOO_BROAD = {
    "machine learning", "deep learning", "artificial intelligence",
    "neural networks", "computer science", "ai", "ml",
}

_embedding_model: SentenceTransformer = None
_topic_anchor_embeddings: np.ndarray = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load embedding model."""
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def _get_topic_anchor_embeddings() -> np.ndarray:
    """Get or compute topic anchor embeddings."""
    global _topic_anchor_embeddings
    if _topic_anchor_embeddings is None:
        settings = get_settings()
        model = _get_embedding_model()
        _topic_anchor_embeddings = model.encode(settings.topic_anchors)
    return _topic_anchor_embeddings


def _check_topic_relevance(query: str) -> float:
    """Check if query is related to AI/ML topics."""
    model = _get_embedding_model()
    query_embedding = model.encode(query)
    anchor_embeddings = _get_topic_anchor_embeddings()

    # Compute cosine similarity with mean of anchors
    mean_anchor = np.mean(anchor_embeddings, axis=0)
    similarity = np.dot(query_embedding, mean_anchor) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(mean_anchor)
    )
    return float(similarity)


def input_guard_node(state: AgentState) -> AgentState:
    """LangGraph node: Input validation and topic filtering."""
    query = state["query"].strip()
    settings = get_settings()

    # 1. Basic safety check
    query_lower = query.lower()
    for phrase in STOP_PHRASES:
        if phrase in query_lower:
            state["error"] = "human_input_required: Query contains potentially harmful content."
            logger.warning(
                "input_guard_blocked_harmful",
                session_id=state["session_id"],
                agent_name="input_guard",
                query=query,
            )
            return state

    # 2. Width check
    if query_lower in TOO_BROAD or len(query.split()) < 3:
        state["error"] = (
            "human_input_required: Query is too broad. "
            "Please provide a more specific topic, e.g., "
            "'retrieval-augmented generation for long-context reasoning' "
            "instead of 'machine learning'."
        )
        logger.warning(
            "input_guard_blocked_broad",
            session_id=state["session_id"],
            agent_name="input_guard",
            query=query,
        )
        return state

    # 3. Topic relevance check
    similarity = _check_topic_relevance(query)
    if similarity < settings.topic_similarity_threshold:
        state["error"] = (
            f"human_input_required: LitRadar is designed for AI/ML research topics. "
            f"Your query appears to be outside this domain (similarity: {similarity:.2f}). "
            f"Please provide a topic related to machine learning, NLP, computer vision, or AI."
        )
        logger.warning(
            "input_guard_blocked_off_topic",
            session_id=state["session_id"],
            agent_name="input_guard",
            query=query,
            similarity=similarity,
        )
        return state

    logger.info(
        "input_guard_passed",
        session_id=state["session_id"],
        agent_name="input_guard",
        query=query,
        similarity=similarity,
    )
    return state


# ============================================================================
# Search Node
# ============================================================================

def search_node(state: AgentState) -> AgentState:
    """LangGraph node: Search ArXiv and Semantic Scholar."""
    if state.get("error"):
        return state

    settings = get_settings()
    subqueries = state.get("subqueries", [])
    papers_seen = state.get("papers_seen", set())

    all_papers = []
    arxiv_available = "arxiv" in state["sources_available"]
    s2_available = "semantic_scholar" in state["sources_available"] and settings.semantic_scholar_enabled

    for subquery in subqueries:
        # Search ArXiv
        if arxiv_available:
            try:
                arxiv_papers = search_arxiv(
                    subquery,
                    max_results=settings.search_results_per_query,
                    papers_seen=papers_seen,
                    session_id=state["session_id"],
                )
                for p in arxiv_papers:
                    if p["paper_id"] not in papers_seen:
                        all_papers.append(p)
                        papers_seen.add(p["paper_id"])
            except Exception as e:
                logger.error(
                    "search_arxiv_failed",
                    session_id=state["session_id"],
                    agent_name="search",
                    subquery=subquery,
                    error=str(e),
                )
                if "arxiv" in state["sources_available"]:
                    state["sources_available"].remove("arxiv")
                    state["source_gap_note"] = "ArXiv was unavailable during search. Results may be incomplete."

        # Search Semantic Scholar (citation-sorted for classic/seminal papers)
        if s2_available:
            try:
                s2_cited_papers = search_semantic_scholar(
                    subquery,
                    max_results=settings.search_results_per_query,
                    papers_seen=papers_seen,
                    session_id=state["session_id"],
                    sort_by_citations=True,
                )
                for p in s2_cited_papers:
                    if p["paper_id"] not in papers_seen:
                        all_papers.append(p)
                        papers_seen.add(p["paper_id"])
            except Exception as e:
                logger.error(
                    "search_s2_failed",
                    session_id=state["session_id"],
                    agent_name="search",
                    subquery=subquery,
                    error=str(e),
                )
                if "semantic_scholar" in state["sources_available"]:
                    state["sources_available"].remove("semantic_scholar")
                    note = "Semantic Scholar was unavailable during search. Results may be incomplete."
                    if state.get("source_gap_note"):
                        state["source_gap_note"] += " " + note
                    else:
                        state["source_gap_note"] = note

    # Check if both sources failed
    if not state["sources_available"]:
        state["error"] = "Both ArXiv and Semantic Scholar are unavailable. Cannot proceed with search."
        return state

    # Check minimum results (only on first iteration)
    if state["iteration"] == 0 and len(all_papers) < 3:
        state["error"] = (
            f"Only found {len(all_papers)} papers. "
            "Please try a broader or different research topic."
        )
        return state

    state["papers_found"] = all_papers
    state["papers_seen"] = papers_seen
    state["total_papers_searched"] += len(all_papers)

    logger.info(
        "search_complete",
        session_id=state["session_id"],
        agent_name="search",
        papers_found=len(all_papers),
        total_papers_searched=state["total_papers_searched"],
        subqueries=len(subqueries),
    )

    return state


# ============================================================================
# Citation Guard
# ============================================================================

def citation_guard_node(state: AgentState) -> AgentState:
    """LangGraph node: Validate citations against accepted papers."""
    if state.get("error"):
        return state

    synthesis = state.get("synthesis_output")
    if not synthesis or not synthesis.get("citations"):
        return state

    # Build valid paper_id set
    valid_ids = set()
    for p in state.get("papers_accepted", []):
        valid_ids.add(p["paper_id"])
    for p in state.get("papers_from_kb", []):
        valid_ids.add(p["paper_id"])

    valid_citations = []
    invalid_citations = []

    for citation in synthesis["citations"]:
        if citation["paper_id"] in valid_ids:
            valid_citations.append(citation)
        else:
            invalid_citations.append(citation)
            logger.warning(
                "citation_guard_rejected",
                session_id=state["session_id"],
                agent_name="citation_guard",
                paper_id=citation["paper_id"],
                claim=citation["claim"],
            )

    # Update synthesis with filtered citations
    state["synthesis_output"]["citations"] = valid_citations
    state["removed_citations"].extend(invalid_citations)

    logger.info(
        "citation_guard_complete",
        session_id=state["session_id"],
        agent_name="citation_guard",
        valid=len(valid_citations),
        removed=len(invalid_citations),
    )

    return state


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_input_guard(state: AgentState) -> Literal["kb_lookup", "__end__"]:
    """Route after input guard."""
    if state.get("error"):
        return END
    return "kb_lookup"


def route_after_sanitize(state: AgentState) -> Literal["critique", "__end__"]:
    """Route after sanitization."""
    if state.get("error"):
        return END

    settings = get_settings()
    if state.get("injection_rate", 0) > settings.injection_rate_stop_threshold:
        state["error"] = (
            "High rate of potential injection patterns detected in search results. "
            "Session stopped for safety."
        )
        logger.security_warning(
            "injection_rate_exceeded",
            session_id=state["session_id"],
            agent_name="sanitizer",
            injection_rate=state["injection_rate"],
        )
        return END

    return "critique"


def route_after_critique(state: AgentState) -> Literal["index", "prepare_iteration", "__end__"]:
    """Route after critic evaluation."""
    if state.get("error"):
        return END

    papers_found = state.get("papers_found", [])
    papers_accepted = state.get("papers_accepted", [])

    # Check if all papers rejected (only on first iteration)
    if state["iteration"] == 0 and not papers_accepted:
        state["error"] = (
            "All papers were rejected as irrelevant. "
            "Please try a different or more specific research topic."
        )
        return END

    # Check high rejection rate (only retry on first iteration)
    if papers_found and state["iteration"] == 0:
        rejection_rate = 1 - (len(papers_accepted) / len(papers_found))
        if rejection_rate > 0.6:
            if state["iteration"] < state["max_iterations"] - 1:
                logger.warning(
                    "high_rejection_rate_retry",
                    session_id=state["session_id"],
                    agent_name="orchestrator",
                    rejection_rate=rejection_rate,
                    iteration=state["iteration"],
                )
                return "prepare_iteration"
            else:
                # Max iterations reached, continue anyway
                logger.warning(
                    "high_rejection_rate_max_iter",
                    session_id=state["session_id"],
                    agent_name="orchestrator",
                    rejection_rate=rejection_rate,
                )

    return "index"


def route_after_synthesis(state: AgentState) -> Literal["citation_guard", "prepare_iteration", "__end__"]:
    """Route after synthesis - reflection loop decision."""
    if state.get("error"):
        return END

    settings = get_settings()

    # Check budget
    if state.get("session_budget_remaining", 1.0) <= 0:
        logger.warning(
            "budget_exceeded",
            session_id=state["session_id"],
            agent_name="orchestrator",
        )
        return "citation_guard"  # Proceed with partial result

    synthesis = state.get("synthesis_output")
    if not synthesis:
        return "citation_guard"

    open_questions = synthesis.get("open_questions", [])
    prev_open_questions = state.get("prev_open_questions", [])

    # Check for reflection loop
    if (
        open_questions
        and open_questions != prev_open_questions
        and state["iteration"] < state["max_iterations"] - 1
    ):
        return "prepare_iteration"

    # Detect loop stall
    if open_questions and open_questions == prev_open_questions:
        logger.warning(
            "loop_stall_detected",
            session_id=state["session_id"],
            agent_name="orchestrator",
        )

    return "citation_guard"


def prepare_iteration_node(state: AgentState) -> AgentState:
    """Prepare state for next iteration (increment counter, save open questions)."""
    synthesis = state.get("synthesis_output") or {}
    open_questions = synthesis.get("open_questions", [])

    state["iteration"] += 1
    state["prev_open_questions"] = open_questions

    logger.info(
        "reflection_loop",
        session_id=state["session_id"],
        agent_name="orchestrator",
        iteration=state["iteration"],
        open_questions=len(open_questions),
    )

    return state


# ============================================================================
# Graph Builder
# ============================================================================

def build_graph() -> StateGraph:
    """Build and compile the LangGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("kb_lookup", kb_lookup_node)
    graph.add_node("plan", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("sanitize", sanitization_node)
    graph.add_node("critique", critic_node)
    graph.add_node("index", retriever_index_node)
    graph.add_node("synthesize", synthesis_node)
    graph.add_node("prepare_iteration", prepare_iteration_node)
    graph.add_node("citation_guard", citation_guard_node)
    graph.add_node("verify", verifier_node)
    graph.add_node("render", renderer_node)

    # Linear edges
    graph.add_edge("kb_lookup", "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "sanitize")
    graph.add_edge("index", "synthesize")
    graph.add_edge("prepare_iteration", "plan")
    graph.add_edge("citation_guard", "verify")
    graph.add_edge("verify", "render")
    graph.add_edge("render", END)

    # Conditional edges
    graph.add_conditional_edges("input_guard", route_after_input_guard)
    graph.add_conditional_edges("sanitize", route_after_sanitize)
    graph.add_conditional_edges("critique", route_after_critique)
    graph.add_conditional_edges("synthesize", route_after_synthesis)

    # Entry point
    graph.set_entry_point("input_guard")

    return graph.compile()


def run_pipeline(query: str) -> AgentState:
    """Run the full LitRadar pipeline."""
    settings = get_settings()

    # Create initial state
    state = create_initial_state(
        query=query,
        max_iterations=settings.max_iterations,
        budget_hard_limit=settings.budget_hard_limit_usd,
    )

    # Initialize Langfuse trace
    llm_client = get_llm_client()
    llm_client.start_trace(state["session_id"], query=query)

    logger.info(
        "session_start",
        session_id=state["session_id"],
        agent_name="orchestrator",
        query=query,
    )

    # Build and run graph
    graph = build_graph()
    final_state = graph.invoke(state)

    # End Langfuse trace with metrics
    llm_client.end_trace(
        session_id=final_state["session_id"],
        output={
            "has_error": bool(final_state.get("error")),
            "papers_accepted": len(final_state.get("papers_accepted", [])),
            "token_count": final_state.get("token_count", 0),
            "iterations": final_state.get("iteration", 0) + 1,
            "total_papers_searched": final_state.get("total_papers_searched", 0),
        },
    )

    logger.info(
        "session_end",
        session_id=final_state["session_id"],
        agent_name="orchestrator",
        has_error=bool(final_state.get("error")),
        papers_accepted=len(final_state.get("papers_accepted", [])),
        token_count=final_state.get("token_count", 0),
    )

    return final_state
