"""LitRadar Streamlit Application."""

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.orchestrator import run_pipeline
from src.config import get_settings


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LitRadar",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 LitRadar")
    st.subheader("AI Research Assistant for Literature Review")

    st.markdown("""
    Enter a research topic in AI/ML to get an automated literature review.
    The system will search ArXiv and Semantic Scholar, evaluate paper relevance,
    identify contradictions, and produce a structured review.
    """)

    # Input
    query = st.text_input(
        "Research Topic",
        placeholder="e.g., chain-of-thought prompting in large language models",
        help="Enter a specific AI/ML research topic. Avoid overly broad terms like 'machine learning'.",
    )

    # Settings in sidebar
    with st.sidebar:
        st.header("Settings")
        settings = get_settings()

        st.markdown(f"**Model:** {settings.llm_model_default}")
        st.markdown(f"**Max Iterations:** {settings.max_iterations}")
        st.markdown(f"**Budget Limit:** ${settings.budget_hard_limit_usd:.2f}")

        st.divider()
        st.markdown("### About")
        st.markdown("""
        LitRadar is an agentic system for automated literature review in AI/ML research.

        **Features:**
        - Multi-source search (ArXiv, Semantic Scholar)
        - Relevance filtering
        - Contradiction detection
        - Citation verification
        - Iterative refinement
        """)

    # Run pipeline
    if st.button("Generate Literature Review", type="primary", disabled=not query):
        if not query.strip():
            st.error("Please enter a research topic.")
            return

        # Progress container
        progress_container = st.container()
        result_container = st.container()

        with progress_container:
            with st.spinner("Starting LitRadar pipeline..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Run pipeline
                status_text.text("🔍 Checking topic and searching knowledge base...")
                progress_bar.progress(10)

                try:
                    final_state = run_pipeline(query)
                    progress_bar.progress(100)

                    # Check for errors
                    if final_state.get("error"):
                        error_msg = final_state["error"]
                        if error_msg.startswith("human_input_required:"):
                            st.warning(error_msg.replace("human_input_required:", "").strip())
                        else:
                            st.error(error_msg)
                        return

                    # Display results
                    with result_container:
                        st.success("✅ Literature review generated successfully!")

                        # Stats
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Papers Searched", final_state.get("total_papers_searched", 0))
                        with col2:
                            st.metric("Papers Accepted", len(final_state.get("papers_accepted", [])))
                        with col3:
                            st.metric("Citations Verified", len(final_state.get("verified_citations", [])))
                        with col4:
                            st.metric("Iterations", final_state.get("iteration", 0) + 1)

                        st.divider()

                        # Main review
                        final_review = final_state.get("final_review")
                        if final_review:
                            st.markdown(final_review)
                        else:
                            st.warning("No review content generated.")

                        # Source gap note
                        if final_state.get("source_gap_note"):
                            st.info(f"⚠️ {final_state['source_gap_note']}")

                        # Removed citations warning
                        removed = final_state.get("removed_citations", [])
                        if removed:
                            with st.expander(f"⚠️ {len(removed)} citation(s) removed during verification"):
                                for c in removed:
                                    st.markdown(f"- **{c['paper_id']}**: {c['claim']}")

                        # Session info
                        st.divider()
                        st.caption(f"Session ID: {final_state.get('session_id', 'N/A')} | Tokens used: {final_state.get('token_count', 0)}")

                except Exception as e:
                    st.error(f"Pipeline failed: {str(e)}")
                    raise


if __name__ == "__main__":
    main()
