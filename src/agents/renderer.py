"""Renderer agent for producing final Markdown review."""

import json
from typing import Optional

from ..state import AgentState
from ..llm_client import get_llm_client
from ..logger import logger

# System prompt - exactly as specified in prompts-renderer.md
RENDERER_SYSTEM_PROMPT = """You are a technical writer producing the final section of an academic literature review.

You will receive a structured JSON summary of a literature review — sections, verified citations, contradictions, and open questions. Your task is to render this into clean, readable Markdown.

## Rules

1. Write in a neutral, academic tone. Third person. Present tense for findings, past tense for what authors did.
2. Every factual claim must be followed by an inline citation in the format [paper_id]. Use only paper_ids from the verified_citations list.
3. Do not add any claims, findings, or paper references that are not in the JSON input. You do not have access to the original abstracts — work only with the summaries provided.
4. Do not invent or paraphrase paper_ids. Copy them exactly as provided.
5. If detected_contradictions is non-empty, include a dedicated "Contradictions and Open Debates" section. Do not bury contradictions inside thematic sections.
6. If open_questions is non-empty, include a dedicated "Open Questions" section at the end.
7. If source_gap_note is provided, include a disclaimer at the top of the review under a "Coverage Note" heading.
8. Keep summaries close to the input — your job is rendering and light editing for flow, not rewriting or expanding.

## Output format

Plain Markdown. Start directly with the first heading. No preamble, no meta-commentary about the task."""


def renderer_node(state: AgentState) -> AgentState:
    """LangGraph node: Renderer agent produces final Markdown."""
    if state.get("error"):
        return state

    synthesis = state.get("synthesis_output")
    if not synthesis:
        state["error"] = "No synthesis output available for rendering."
        return state

    client = get_llm_client()

    # Build synthesis JSON for prompt
    # Use verified_citations instead of original citations
    render_data = {
        "sections": synthesis.get("sections", []),
        "verified_citations": state.get("verified_citations", []),
        "detected_contradictions": synthesis.get("detected_contradictions", []),
        "open_questions": synthesis.get("open_questions", []),
    }

    synthesis_json = json.dumps(render_data, indent=2, ensure_ascii=False)
    source_gap_note = state.get("source_gap_note") or "null"

    user_prompt = f"""Research topic: {state['query']}

## Synthesis JSON

{synthesis_json}

The synthesis_json contains:
- sections: thematic groups with summaries and paper_ids
- verified_citations: [{{paper_id, claim}}] — the only citations you may use
- detected_contradictions: [{{topic, paper_a, paper_b, description}}]
- open_questions: [string]

## Source coverage note

{source_gap_note}

If this field is not null, include it as a disclaimer at the top.

## Task

Render the literature review in Markdown. Follow the structure below.

---

# Literature Review: {state['query']}

{{coverage_note_if_present}}

## {{section_title_1}}

{{section_summary with inline [paper_id] citations}}

## {{section_title_2}}

...

## Contradictions and Open Debates

{{one paragraph per contradiction, describing the positions of paper_a and paper_b}}

## Open Questions

{{bulleted list of open questions}}"""

    messages = [
        {"role": "system", "content": RENDERER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    retries = 3
    for attempt in range(retries):
        try:
            content, input_tokens, output_tokens = client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                session_id=state["session_id"],
                agent_name="renderer",
            )

            state["token_count"] += input_tokens + output_tokens

            # Validate output
            if len(content.strip()) < 100:
                logger.warning(
                    "renderer_output_too_short",
                    session_id=state["session_id"],
                    agent_name="renderer",
                    attempt=attempt + 1,
                    length=len(content),
                )
                if attempt < retries - 1:
                    continue

            # Check if output contains JSON fragments instead of Markdown
            if content.strip().startswith("{") or '"sections"' in content[:100]:
                logger.warning(
                    "renderer_output_json_instead_of_markdown",
                    session_id=state["session_id"],
                    agent_name="renderer",
                    attempt=attempt + 1,
                )
                if attempt < retries - 1:
                    continue

            state["final_review"] = content

            logger.info(
                "renderer_complete",
                session_id=state["session_id"],
                agent_name="renderer",
                length=len(content),
            )
            return state

        except Exception as e:
            logger.error(
                "renderer_attempt_failed",
                session_id=state["session_id"],
                agent_name="renderer",
                attempt=attempt + 1,
                error=str(e),
            )

    # Fallback: return raw synthesis JSON
    logger.error(
        "renderer_failed_all_retries",
        session_id=state["session_id"],
        agent_name="renderer",
    )
    state["final_review"] = f"""# Synthesis output (rendering failed)

```json
{synthesis_json}
```
"""

    return state
