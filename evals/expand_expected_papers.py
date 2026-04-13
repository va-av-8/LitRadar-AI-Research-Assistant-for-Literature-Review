"""Expand expected_papers in test_topics.json via Semantic Scholar API."""

import json
import time
from pathlib import Path
from typing import List, Set

import httpx


def search_s2(query: str, limit: int = 30) -> List[dict]:
    """Search Semantic Scholar for papers."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "paperId,externalIds,title,year,citationCount,abstract",
        "limit": limit,
    }

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url, params=params)
            if response.status_code == 429:
                print("  [RATE LIMITED] Waiting 60s...")
                time.sleep(60)
                response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []


def filter_papers(
    papers: List[dict],
    existing_ids: Set[str],
    min_citations: int = 100,
) -> List[dict]:
    """Filter papers by criteria."""
    filtered = []
    for p in papers:
        # Must have arxiv ID
        ext_ids = p.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        if not arxiv_id:
            continue

        # Format ID
        paper_id = f"arxiv:{arxiv_id}"

        # Skip duplicates
        if paper_id.lower() in existing_ids:
            continue

        # Must have abstract
        if not p.get("abstract"):
            continue

        # Citation threshold
        citations = p.get("citationCount") or 0
        if citations < min_citations:
            continue

        filtered.append({
            "id": paper_id,
            "title": p.get("title", ""),
            "citations": citations,
        })
        existing_ids.add(paper_id.lower())

    # Sort by citations descending
    filtered.sort(key=lambda x: x["citations"], reverse=True)
    return filtered


def expand_topic(topic: dict) -> tuple[int, int]:
    """Expand expected_papers for a single topic. Returns (before, after) counts."""
    query = topic["query"]
    existing = set(p.lower() for p in topic["expected_papers"])
    before_count = len(topic["expected_papers"])
    new_papers = []

    # Query 1: Main topic
    print(f"  Searching: {query[:50]}...")
    time.sleep(1)
    results = search_s2(query)
    filtered = filter_papers(results, existing, min_citations=100)
    new_papers.extend(filtered[:15])

    # Query 2: Survey/review papers
    survey_query = f"{query} survey OR review"
    print(f"  Searching: {survey_query[:50]}...")
    time.sleep(1)
    results = search_s2(survey_query)
    filtered = filter_papers(results, existing, min_citations=100)
    new_papers.extend(filtered[:10])

    # If too few papers found, lower threshold
    if len(new_papers) < 5:
        print(f"  [WARNING] Only {len(new_papers)} papers found, lowering threshold to 50")
        time.sleep(1)
        results = search_s2(query)
        filtered = filter_papers(results, existing, min_citations=50)
        new_papers.extend(filtered[:15 - len(new_papers)])

    # Take up to 15 total new papers
    new_papers = new_papers[:15]

    # Add to expected_papers
    for p in new_papers:
        topic["expected_papers"].append(p["id"])
        print(f"    + {p['id']} ({p['citations']} cites) - {p['title'][:50]}...")

    after_count = len(topic["expected_papers"])
    return before_count, after_count


def main():
    topics_path = Path(__file__).parent / "test_topics.json"

    with open(topics_path) as f:
        topics = json.load(f)

    print("Expanding expected_papers via Semantic Scholar API\n")
    print("=" * 60)

    summary = []

    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] {topic['query']}")
        before, after = expand_topic(topic)
        diff = after - before
        summary.append((topic["query"], before, after, diff))
        print(f"  Result: {before} → {after} papers (+{diff})")

    # Save updated JSON
    with open(topics_path, "w") as f:
        json.dump(topics, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for query, before, after, diff in summary:
        short_query = query[:45] + "..." if len(query) > 45 else query
        print(f"{short_query}: {before} → {after} papers (+{diff})")

    total_before = sum(s[1] for s in summary)
    total_after = sum(s[2] for s in summary)
    print(f"\nTotal: {total_before} → {total_after} papers (+{total_after - total_before})")
    print(f"\nUpdated: {topics_path}")


if __name__ == "__main__":
    main()
