"""ChromaDB-based retriever for knowledge base operations."""

from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .config import get_settings
from .state import PaperMetadata, AgentState
from .logger import logger


class Retriever:
    """ChromaDB wrapper for KB lookup, indexing, and RAG retrieval."""

    def __init__(self):
        self.settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=self.settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_model: Optional[SentenceTransformer] = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
        return self._embedding_model

    def _get_text_for_embedding(self, paper: PaperMetadata) -> str:
        """Get text to embed for a paper."""
        if paper.get("abstract"):
            return paper["abstract"]
        # Fallback for missing abstract
        authors = ", ".join(paper.get("authors", []))
        return f"{paper['title']}. {authors}"

    def kb_lookup(self, query: str, top_k: Optional[int] = None) -> List[PaperMetadata]:
        """
        Retrieve papers from KB by query similarity.

        Called at session start before Planner.
        """
        top_k = top_k or self.settings.kb_lookup_top_k

        # Check if collection is empty
        if self._collection.count() == 0:
            return []

        # Embed query
        query_embedding = self.embedding_model.encode(query).tolist()

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["metadatas", "documents"],
        )

        papers: List[PaperMetadata] = []
        if results["metadatas"] and results["metadatas"][0]:
            for metadata in results["metadatas"][0]:
                papers.append(PaperMetadata(
                    paper_id=metadata.get("paper_id", ""),
                    title=metadata.get("title", ""),
                    authors=metadata.get("authors", "").split("; ") if metadata.get("authors") else [],
                    abstract=metadata.get("abstract"),
                    year=int(metadata["year"]) if metadata.get("year") else None,
                    source=metadata.get("source", ""),
                    citation_count=int(metadata["citation_count"]) if metadata.get("citation_count") else None,
                ))

        return papers

    def index(self, papers: List[PaperMetadata], session_id: str = "") -> int:
        """
        Index new papers into ChromaDB.

        Returns number of papers actually added (skips duplicates).
        """
        if not papers:
            return 0

        # Get existing paper_ids
        existing_ids = set()
        if self._collection.count() > 0:
            all_results = self._collection.get(include=[])
            existing_ids = set(all_results["ids"])

        # Filter out duplicates
        new_papers = [p for p in papers if p["paper_id"] not in existing_ids]

        if not new_papers:
            logger.info(
                "index_skipped_all_duplicates",
                session_id=session_id,
                agent_name="retriever",
                total=len(papers),
            )
            return 0

        # Prepare data for indexing
        ids = [p["paper_id"] for p in new_papers]
        documents = [self._get_text_for_embedding(p) for p in new_papers]
        embeddings = self.embedding_model.encode(documents).tolist()

        metadatas = []
        for p in new_papers:
            metadatas.append({
                "paper_id": p["paper_id"],
                "title": p["title"],
                "authors": "; ".join(p.get("authors", [])),
                "abstract": p.get("abstract") or "",
                "year": p.get("year") or 0,
                "source": p["source"],
                "citation_count": p.get("citation_count") or 0,
            })

        # Add to collection
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "papers_indexed",
            session_id=session_id,
            agent_name="retriever",
            indexed=len(new_papers),
            skipped=len(papers) - len(new_papers),
        )

        return len(new_papers)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[PaperMetadata]:
        """
        RAG retrieval for Synthesis.

        Returns top-k papers by similarity to query.
        """
        top_k = top_k or self.settings.retriever_top_k_synthesis

        if self._collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["metadatas", "documents"],
        )

        papers: List[PaperMetadata] = []
        if results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                papers.append(PaperMetadata(
                    paper_id=metadata.get("paper_id", ""),
                    title=metadata.get("title", ""),
                    authors=metadata.get("authors", "").split("; ") if metadata.get("authors") else [],
                    abstract=metadata.get("abstract"),
                    year=int(metadata["year"]) if metadata.get("year") else None,
                    source=metadata.get("source", ""),
                    citation_count=int(metadata["citation_count"]) if metadata.get("citation_count") else None,
                ))

        return papers


# Global retriever instance
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get or create retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def reset_retriever() -> None:
    """Reset retriever singleton (call after clearing ChromaDB).

    Note: For full DB reset, call client.reset() BEFORE this function.
    This function only clears the Python singleton reference.
    """
    global _retriever
    if _retriever is not None:
        try:
            # Clear references to allow garbage collection
            if _retriever._embedding_model is not None:
                _retriever._embedding_model = None
            _retriever._collection = None
            _retriever._client = None
        except Exception:
            pass
    _retriever = None
    # Force garbage collection to release file handles
    import gc
    gc.collect()


def kb_lookup_node(state: AgentState) -> AgentState:
    """LangGraph node: KB lookup at session start."""
    retriever = get_retriever()
    papers = retriever.kb_lookup(state["query"])
    state["papers_from_kb"] = papers

    logger.info(
        "kb_lookup",
        session_id=state["session_id"],
        agent_name="retriever",
        papers_from_kb=len(papers),
    )

    return state


def retriever_index_node(state: AgentState) -> AgentState:
    """LangGraph node: Index papers_accepted after Critic."""
    if state.get("error"):
        return state

    retriever = get_retriever()
    indexed = retriever.index(state["papers_accepted"], state["session_id"])

    logger.info(
        "papers_indexed",
        session_id=state["session_id"],
        agent_name="retriever",
        papers_indexed=indexed,
    )

    return state
