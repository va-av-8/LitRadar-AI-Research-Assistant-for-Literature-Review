"""Evaluation runner for LitRadar."""

import os
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.orchestrator import run_pipeline
from src.retriever import get_retriever, reset_retriever
from src.state import PaperMetadata
from src.config import get_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_topics(path: str) -> List[Dict[str, Any]]:
    """Load test topics from JSON file."""
    with open(path) as f:
        return json.load(f)


def save_partial(results: List[Dict[str, Any]], output_path: Path, timestamp: str) -> None:
    """Save partial results after each topic so progress is not lost on crash."""
    partial_file = output_path / f"eval_results_{timestamp}.json"
    with open(partial_file, "w") as f:
        json.dump(results, f, indent=2)


def reset_chroma_db() -> None:
    """Clear ChromaDB data via API to ensure clean state between topics.

    Deletes all collections via API, then resets the singleton.
    This avoids SQLite file handle issues from file deletion.
    """
    try:
        retriever = get_retriever()
        if retriever._client is not None:
            # Delete all collections via API (doesn't require allow_reset)
            for col in retriever._client.list_collections():
                retriever._client.delete_collection(col.name)
    except Exception as e:
        print(f"[WARN] ChromaDB collection cleanup failed: {e}")

    # Reset singleton - next get_retriever() creates fresh client with new collection
    reset_retriever()
    time.sleep(0.2)


def fetch_top_cited_from_openalex(
    query: str,
    min_citations: int = 50,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """
    Fetch top cited papers from OpenAlex for a given query.

    Strategy: filter by AI/ML topics for relevance, fetch results sorted
    by relevance, then sort by citations locally.

    Filters:
    - topics.id: NLP, Neural Networks, CV topics (OR)
    - indexed_in:arxiv (must have arxiv ID)
    - cited_by_count >= min_citations
    - publication_year >= 2020 (focus on recent AI/ML work)

    Returns list of dicts with: arxiv_id, title, citation_count, abstract
    """
    url = "https://api.openalex.org/works"
    # Fetch more results (up to 100) sorted by relevance, then filter and pick top-cited
    fetch_count = min(limit * 7, 100)

    # AI/ML topic IDs:
    # T10181 = Natural Language Processing Techniques
    # T10320 = Neural Networks and Applications
    # T10036 = Advanced Neural Network Applications (CV, etc.)
    # T12549 = Image and Object Detection Techniques
    topics_filter = "topics.id:T10181|T10320|T10036|T12549"

    params = {
        "search": query,
        "filter": f"{topics_filter},indexed_in:arxiv,cited_by_count:>{min_citations},publication_year:>2020",
        # Default sort is by relevance - don't override with cited_by_count
        "per_page": fetch_count,
        "select": "id,title,authorships,publication_year,cited_by_count,abstract_inverted_index,ids,locations",
    }

    # Extract keywords from query for local relevance filtering
    query_keywords = set(word.lower() for word in query.split() if len(word) > 3)

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        for work in data.get("results", []):
            # Extract arxiv ID from locations (not from ids - OpenAlex doesn't put it there)
            arxiv_id = None
            for location in work.get("locations", []):
                landing_url = location.get("landing_page_url", "")
                if "arxiv.org/abs/" in landing_url:
                    arxiv_id = landing_url.split("arxiv.org/abs/")[-1]
                    break

            if not arxiv_id:
                continue

            # Reconstruct abstract from inverted index
            abstract = None
            inverted = work.get("abstract_inverted_index")
            if inverted:
                # Inverted index: {"word": [positions], ...}
                words = [""] * (max(max(pos) for pos in inverted.values()) + 1)
                for word, positions in inverted.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words)

            # Extract authors
            authors = []
            for authorship in work.get("authorships", [])[:10]:  # Limit authors
                author = authorship.get("author", {})
                name = author.get("display_name")
                if name:
                    authors.append(name)

            title = work.get("title", "") or ""
            results.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "year": work.get("publication_year"),
                "citation_count": work.get("cited_by_count", 0),
            })

        # Filter by keyword relevance: title or abstract must contain query keywords
        def is_relevant(paper):
            text = (paper["title"] + " " + (paper["abstract"] or "")).lower()
            matches = sum(1 for kw in query_keywords if kw in text)
            return matches >= min(2, len(query_keywords))  # At least 2 keywords or all if fewer

        relevant_results = [p for p in results if is_relevant(p)]

        # If too few relevant results, fall back to all results
        if len(relevant_results) < limit // 2:
            relevant_results = results

        # Sort by citation count and return top N
        relevant_results.sort(key=lambda x: x["citation_count"], reverse=True)
        return relevant_results[:limit]

    except Exception as e:
        print(f"[OpenAlex API error: {e}]")
        return []


def preload_top_cited_papers(query: str, min_citations: int = 50, limit: int = 25) -> int:
    """
    Pre-load top cited papers from OpenAlex into ChromaDB.

    This provides "background knowledge" of classic papers without
    loading the exact ground truth papers (avoiding data leakage).

    Returns count of successfully loaded papers.
    """
    print(f"  Fetching top cited papers from OpenAlex...")
    papers_data = fetch_top_cited_from_openalex(query, min_citations, limit)

    if not papers_data:
        print(f"  [WARN] No papers found from OpenAlex")
        return 0

    print(f"  Found {len(papers_data)} papers from OpenAlex")

    retriever = get_retriever()
    papers_to_index: List[PaperMetadata] = []

    for p in papers_data:
        if not p.get("abstract"):
            continue

        paper = PaperMetadata(
            paper_id=f"arxiv:{p['arxiv_id']}",
            title=p["title"],
            authors=p.get("authors", []),
            abstract=p["abstract"],
            year=p.get("year"),
            source="arxiv",
            citation_count=p.get("citation_count"),
        )
        papers_to_index.append(paper)
        print(f"    + arxiv:{p['arxiv_id']} ({p['citation_count']} cites) - {p['title'][:45]}...")

    if papers_to_index:
        indexed = retriever.index(papers_to_index, session_id="eval_preload_openalex")
        print(f"  Indexed {indexed} top cited papers into ChromaDB")
        return indexed

    return 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize_paper_id(pid: str) -> str:
    """Normalize paper ID to bare arxiv ID format for comparison."""
    pid = pid.strip()
    for prefix in ["arxiv:", "s2:", "https://arxiv.org/abs/", "http://arxiv.org/abs/"]:
        if pid.lower().startswith(prefix.lower()):
            pid = pid[len(prefix):]
    # Remove version suffix (e.g., v1, v2)
    if "v" in pid:
        pid = pid.rsplit("v", 1)[0]
    return pid.lower()


def compute_coverage(found_paper_ids: List[str], expected_papers: List[str]) -> float:
    """Coverage = |found ∩ expected| / |expected|."""
    if not expected_papers:
        return 1.0
    found_set = set(normalize_paper_id(p) for p in found_paper_ids)
    expected_set = set(normalize_paper_id(p) for p in expected_papers)
    return len(found_set & expected_set) / len(expected_set)


def compute_hallucination_rate(removed: int, verified: int) -> Optional[float]:
    """Hallucination Rate = removed_citations / total_citations."""
    total = removed + verified
    if total == 0:
        return None
    return removed / total


def compute_contradiction_recall(
    detected_contradictions: List[Dict],
    expected_pairs: List[List[str]],
) -> Optional[float]:
    """
    Contradiction Recall = expected pairs covered / total expected pairs.

    A pair is covered if both paper_ids appear together in any detected contradiction.
    """
    if not expected_pairs:
        return None

    detected_pairs: List[frozenset] = [
        frozenset([c.get("paper_a", ""), c.get("paper_b", "")])
        for c in detected_contradictions
    ]

    covered = 0
    for pair in expected_pairs:
        pair_set = frozenset(pair[:2])  # first two elements are paper_ids
        if pair_set in detected_pairs:
            covered += 1

    return covered / len(expected_pairs)


# ---------------------------------------------------------------------------
# Single-topic eval
# ---------------------------------------------------------------------------

def run_eval(test_topic: Dict[str, Any], preload: bool = True) -> Dict[str, Any]:
    """Run evaluation for a single topic and return result dict."""
    query = test_topic["query"]
    expected_papers = test_topic.get("expected_papers", [])
    should_have_contradictions = test_topic.get("should_have_contradictions", False)
    contradiction_pairs = test_topic.get("contradiction_pairs", [])

    print(f"\n{'='*60}")
    print(f"Topic: {query}")
    print(f"Expected papers: {len(expected_papers)}")
    print(f"{'='*60}")

    # Pre-load top cited papers from OpenAlex (background knowledge, not ground truth)
    if preload:
        print(f"\nPre-loading top cited papers from OpenAlex...")
        loaded = preload_top_cited_papers(query, min_citations=50, limit=25)
        print(f"Pre-loaded: {loaded} top cited papers\n")

    try:
        final_state = run_pipeline(query)

        # --- Papers ---
        papers_accepted = final_state.get("papers_accepted", [])
        papers_from_kb = final_state.get("papers_from_kb", [])
        papers_found = final_state.get("papers_found", [])

        # Coverage includes both accepted in this session AND papers retrieved from KB
        all_found_ids = (
            [p["paper_id"] for p in papers_accepted]
            + [p["paper_id"] for p in papers_from_kb]
        )

        # Retrieval coverage: papers found BEFORE Critic filter
        papers_found_ids = [p["paper_id"] for p in papers_found]

        # DEBUG: Check paper ID formats
        if all_found_ids:
            print(f"[DEBUG] Sample found IDs (accepted): {all_found_ids[:5]}")
            print(f"[DEBUG] Sample expected IDs: {expected_papers[:5]}")
        if papers_found_ids:
            print(f"[DEBUG] Sample found IDs (before critic): {papers_found_ids[:5]}")

        # --- Synthesis ---
        # synthesis_output is Optional — guard against None
        synthesis = final_state.get("synthesis_output") or {}
        detected_contradictions = synthesis.get("detected_contradictions", [])
        open_questions = synthesis.get("open_questions", [])

        # --- Citations ---
        verified = len(final_state.get("verified_citations", []))
        removed = len(final_state.get("removed_citations", []))

        # --- Metrics ---
        retrieval_coverage = compute_coverage(papers_found_ids, expected_papers)
        accepted_coverage = compute_coverage(all_found_ids, expected_papers)
        hallucination_rate = compute_hallucination_rate(removed, verified)
        has_contradictions = len(detected_contradictions) > 0
        contradiction_recall = compute_contradiction_recall(
            detected_contradictions, contradiction_pairs
        )

        # Open Questions Resolution requires iter-1 open questions stored in state;
        # AgentState tracks prev_open_questions — use it if available.
        prev_open_questions = final_state.get("prev_open_questions", [])
        if prev_open_questions:
            resolved = len(prev_open_questions) - len(open_questions)
            oq_resolution = max(resolved, 0) / len(prev_open_questions)
        else:
            oq_resolution = None  # single iteration — metric not applicable

        result = {
            "query": query,
            "status": "success" if not final_state.get("error") else "error",
            "error": final_state.get("error"),

            # Paper counts
            "papers_found": len(final_state.get("papers_found", [])),
            "papers_accepted": len(papers_accepted),
            "papers_from_kb": len(papers_from_kb),

            # Coverage
            "retrieval_coverage": retrieval_coverage,
            "accepted_coverage": accepted_coverage,
            "coverage_passed": accepted_coverage >= 0.7,

            # Citations
            "verified_citations": verified,
            "removed_citations": removed,
            "hallucination_rate": hallucination_rate,
            "hallucination_passed": hallucination_rate == 0.0 if hallucination_rate is not None else None,

            # Contradictions
            "contradictions_found": len(detected_contradictions),
            "has_contradictions": has_contradictions,
            "contradictions_expected": should_have_contradictions,
            "contradictions_correct": has_contradictions == should_have_contradictions,
            "contradiction_recall": contradiction_recall,

            # Open questions
            "open_questions_count": len(open_questions),
            "open_questions_resolution": oq_resolution,
            "oq_resolution_passed": oq_resolution >= 0.6 if oq_resolution is not None else None,

            # Session
            "iterations": final_state.get("iteration", 0) + 1,
            "token_count": final_state.get("token_count", 0),
        }

        # Print summary
        print(f"Status: {result['status']}")
        print(f"Papers: {result['papers_accepted']} accepted / {result['papers_found']} found / {result['papers_from_kb']} from KB")
        print(f"Retrieval coverage: {result['retrieval_coverage']:.2%} (before Critic)")
        print(f"Accepted coverage: {result['accepted_coverage']:.2%} — {'PASS' if result['coverage_passed'] else 'FAIL'}")
        if hallucination_rate is not None:
            print(f"Hallucination rate: {hallucination_rate:.2%} — {'PASS' if result['hallucination_passed'] else 'FAIL'}")
        print(f"Contradictions: {result['contradictions_found']} found, expected={should_have_contradictions} — {'PASS' if result['contradictions_correct'] else 'FAIL'}")
        if contradiction_recall is not None:
            print(f"Contradiction recall: {contradiction_recall:.2%}")
        if oq_resolution is not None:
            print(f"Open questions resolution: {oq_resolution:.2%} — {'PASS' if result['oq_resolution_passed'] else 'FAIL'}")

        return result

    except Exception as e:
        import traceback
        print(f"Exception: {e}")
        traceback.print_exc()
        return {
            "query": query,
            "status": "exception",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Full eval run
# ---------------------------------------------------------------------------

def run_all_evals(topics_path: str, output_dir: str, reset_db: bool = False, preload: bool = True) -> None:
    """Run evaluations for all test topics."""
    test_topics = load_test_topics(topics_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for i, topic in enumerate(test_topics, 1):
        print(f"\n[{i}/{len(test_topics)}] {topic['query']}")

        if reset_db:
            print("Resetting ChromaDB...")
            reset_chroma_db()

        result = run_eval(topic, preload=preload)
        results.append(result)

        # Save after each topic — progress is never lost
        save_partial(results, output_path, timestamp)

    # Final summary
    summary = generate_summary(results)
    summary_file = output_path / f"eval_summary_{timestamp}.md"
    with open(summary_file, "w") as f:
        f.write(summary)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"Results: {output_path / f'eval_results_{timestamp}.json'}")
    print(f"Summary: {summary_file}")
    print(f"{'='*60}")
    print(summary)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def generate_summary(results: List[Dict[str, Any]]) -> str:
    """Generate Markdown summary of evaluation results."""
    total = len(results)
    successful = [r for r in results if r["status"] == "success"]
    n = len(successful)

    def avg(key: str) -> Optional[float]:
        vals = [r[key] for r in successful if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def pct(key: str, threshold: bool = True) -> str:
        count = sum(1 for r in successful if r.get(key) == threshold)
        return f"{count}/{n} ({100*count/n:.0f}%)" if n else "N/A"

    avg_retrieval_coverage = avg("retrieval_coverage")
    avg_accepted_coverage = avg("accepted_coverage")
    avg_hallucination = avg("hallucination_rate")
    avg_contradiction_recall = avg("contradiction_recall")
    avg_oq_resolution = avg("open_questions_resolution")
    avg_tokens = avg("token_count")

    lines = [
        "# LitRadar Evaluation Summary",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Topics evaluated:** {total} | **Successful:** {n}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value | Target |",
        "|--------|-------|--------|",
        f"| Accepted coverage ≥ 70% | {pct('coverage_passed')} | ≥ 70% of topics |",
        f"| Avg retrieval coverage | {avg_retrieval_coverage:.2%} | — |" if avg_retrieval_coverage else "| Avg retrieval coverage | N/A | — |",
        f"| Avg accepted coverage | {avg_accepted_coverage:.2%} | ≥ 70% |" if avg_accepted_coverage else "| Avg accepted coverage | N/A | ≥ 70% |",
        f"| Hallucination rate = 0% | {pct('hallucination_passed')} | 100% of topics |",
        f"| Average hallucination rate | {avg_hallucination:.2%} | 0% |" if avg_hallucination is not None else "| Average hallucination rate | N/A | 0% |",
        f"| Contradictions correct | {pct('contradictions_correct')} | — |",
        f"| Avg contradiction recall | {avg_contradiction_recall:.2%} | ≥ 60% |" if avg_contradiction_recall else "| Avg contradiction recall | N/A | ≥ 60% |",
        f"| OQ resolution ≥ 60% | {pct('oq_resolution_passed')} | ≥ 60% of topics |",
        f"| Avg OQ resolution | {avg_oq_resolution:.2%} | ≥ 60% |" if avg_oq_resolution else "| Avg OQ resolution | N/A | ≥ 60% |",
        f"| Average tokens/session | {avg_tokens:.0f} | — |" if avg_tokens else "| Average tokens/session | N/A | — |",
        "",
        "## Per-Topic Results",
        "",
        "| Topic | Status | Retr. Cov | Acc. Cov | Halluc. | Contr. recall | OQ res. | Tokens |",
        "|-------|--------|-----------|----------|---------|---------------|---------|--------|",
    ]

    for r in results:
        query = r["query"][:38] + ".." if len(r["query"]) > 40 else r["query"]
        status = {"success": "✅", "error": "⚠️", "exception": "❌"}.get(r.get("status", ""), "❓")
        retr_coverage = f"{r['retrieval_coverage']:.0%}" if r.get("retrieval_coverage") is not None else "N/A"
        acc_coverage = f"{r['accepted_coverage']:.0%}" if r.get("accepted_coverage") is not None else "N/A"
        hallucination = f"{r['hallucination_rate']:.2%}" if r.get("hallucination_rate") is not None else "N/A"
        c_recall = f"{r['contradiction_recall']:.0%}" if r.get("contradiction_recall") is not None else "N/A"
        oq = f"{r['open_questions_resolution']:.0%}" if r.get("open_questions_resolution") is not None else "N/A"
        tokens = str(r.get("token_count", "N/A"))
        lines.append(f"| {query} | {status} | {retr_coverage} | {acc_coverage} | {hallucination} | {c_recall} | {oq} | {tokens} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LitRadar Evaluation Runner")
    parser.add_argument(
        "--topics",
        default="evals/test_topics.json",
        help="Path to test topics JSON file",
    )
    parser.add_argument(
        "--output",
        default="evals/results/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset ChromaDB between topics for isolated per-topic evaluation",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip preloading expected papers into ChromaDB",
    )

    args = parser.parse_args()
    run_all_evals(args.topics, args.output, reset_db=args.reset_db, preload=not args.no_preload)


if __name__ == "__main__":
    main()
