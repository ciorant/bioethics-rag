import faiss
import numpy as np
import pickle
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from threading import Lock
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


DocumentChunk = Dict[str, Any]

class FAISSVectorStore:
    def __init__(
        self,
        dimension: int = 3072,
        index_path: str = "data/faiss_index",
        embedding_model: str = "text-embedding-3-large", #3072-dim vectors
    ):
        if OpenAIEmbeddings is None:
            raise ImportError(
                "Could not import OpenAIEmbeddings from langchain. "
                "Install langchain or adapt the import to your environment."
            )

        self.dimension = dimension
        self.index_path = Path(index_path)
        self._lock = Lock()
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Instantiate embeddings (may make API calls later when embedding)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # in-memory structures
        self.documents: List[DocumentChunk] = []

        # Create a new FAISS index (will be replaced by load if a saved index exists)
        self.index = faiss.IndexFlatIP(self.dimension) # All vectors must be this length

        # If there's a saved index, load it (overwrites the index created above).
        self.load_index()  # safe: will return False if nothing to load

    def _ensure_index_dim(self, d: int):
        """Ensure FAISS index has dimension d."""
        # If current index has no vectors, and d != self.dimension, recreate.
        # Using getattr for defensive programming
        if getattr(self.index, "ntotal", 0) == 0 and getattr(self.index, "d", None) != d:
            logger.info("Recreating an empty index with dimension %d", d)
            self.dimension = d
            self.index = faiss.IndexFlatIP(self.dimension)
        elif getattr(self.index, "d", None) is not None and self.index.d != d:
            raise ValueError(f"Embedding dimension ({d}) does not match existing index dimension ({self.index.d}).")

    def add_documents(self, chunks: List[DocumentChunk], save: bool = True):
        """
        Add list of chunks to the FAISS index. Each chunk MUST contain 'text'.
        If index is empty and embedding dimension differs, the index will be re-created.
        """
        with self._lock:
            if not chunks:
                logger.debug("No chunks to add.")
                return

            texts = []
            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    raise ValueError(f"Chunk {i} is not a dictionary")
                if "text" not in chunk:
                    raise ValueError(f"Chunk {i} missing required 'text' field")
                if not isinstance(chunk["text"], str):
                    raise ValueError(f"Chunk {i} 'text' field must be a string")
                if not chunk["text"].strip():
                    logger.warning(f"Chunk {i} has empty text content")
                    continue
                texts.append(chunk["text"])

            # Get embeddings from the embedding provider (call to a model)
            embeddings = self.embeddings.embed_documents(texts)
            embeddings_np = np.asarray(embeddings, dtype=np.float32)

            # Embedding shape checks
            if embeddings_np.ndim == 1:
                # single vector returned as 1D array -> reshape to (1, d)
                embeddings_np = embeddings_np.reshape(1, -1)

            emb_d = embeddings_np.shape[1]
            # If needed, recreate the index dimension (only possible if index currently empty)
            self._ensure_index_dim(emb_d)

            if emb_d != self.index.d:
                raise ValueError(f"Embedding dim {emb_d} != index dim {self.index.d}")

            # L2-normalize rows (in place) so inner product == cosine similarity
            faiss.normalize_L2(embeddings_np)

            # Add to index
            self.index.add(embeddings_np)
            # The documentation of "add" suggests we have to put the number of vectors,
            # as a first argument, but Python does it for us.

            # Append documents (simple positional mapping: index position -> documents list)
            self.documents.extend(chunks)

            if save:
                self.save_index()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search similar documents for `query`. Returns up to k results.
        Each result: { "content": <text>, "metadata": <metadata>, "similarity_score": <float> }
        similarity_score is the inner product of normalized vectors => cosine similarity in [-1,1].
        """
        with self._lock:
            # guard: no vectors at all
            if getattr(self.index, "ntotal", 0) == 0:
                logger.debug("Search called but index is empty.")
                return []

            # embed query
            q_emb = self.embeddings.embed_query(query)
            q_np = np.asarray([q_emb], dtype=np.float32)
            if q_np.ndim == 1:
                q_np = q_np.reshape(1, -1)

            if q_np.shape[1] != self.index.d:
                # if index is empty we could recreate; but at this point we know index has vectors.
                raise ValueError(f"Query embedding dim {q_np.shape[1]} does not match index dimension {self.index.d}")

            faiss.normalize_L2(q_np)

            # clamp k
            k = min(k, int(self.index.ntotal))

            distances, indices = self.index.search(q_np, k)  # distances shape (1,k) ; indices shape (1,k)

            results = []
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    # FAISS returns -1 for "empty" slots sometimes; skip
                    continue
                if idx >= len(self.documents):
                    logger.warning("Index returned idx %d but documents list has length %d", idx, len(self.documents))
                    continue
                doc = self.documents[idx]
                results.append({
                    "content": doc.get("text"),
                    "metadata": doc.get("metadata", {}),
                    "similarity_score": float(score)  # already cosine because of normalization
                })
            return results

    def save_index(self):
        """Persist index and documents to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        logger.debug("FAISS index and documents saved to %s", self.index_path)

    def load_index(self) -> bool:
        """Load index and documents from disk. Returns True if loaded."""
        index_file = self.index_path / "index.faiss"
        docs_file = self.index_path / "documents.pkl"

        if index_file.exists() and docs_file.exists():
            self.index = faiss.read_index(str(index_file))
            with open(docs_file, "rb") as f:
                self.documents = pickle.load(f)

            # update dimension to match loaded index
            if getattr(self.index, "d", None) is not None:
                self.dimension = int(self.index.d)

            if self.index.d == 0 or len(self.documents) != self.index.ntotal:
                logger.error("Corrupted index detected, deleting...")
                index_file.unlink()
                docs_file.unlink()
                return False

            # warn if counts differ
            if len(self.documents) != self.index.ntotal:
                logger.warning(
                    "Loaded documents list length (%d) differs from index.ntotal (%d). "
                    "This can lead to mismatches. Using what's available.",
                    len(self.documents),
                    self.index.ntotal,
                )
            logger.info("Loaded FAISS index from %s (ntotal=%d, dim=%d)",
                        index_file, int(self.index.ntotal), int(self.index.d))
            return True
        return False

