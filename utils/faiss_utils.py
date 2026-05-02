"""utils/faiss_utils.py — FAISS index loading and search."""

import numpy as np


def load_index(index_path: str):
    """Load a FAISS index from disk."""
    try:
        import faiss
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu")

    index = faiss.read_index(index_path)
    print(f"[FAISS] Loaded index: {index.ntotal:,} vectors, dim={index.d}")
    return index


def query_index(index, embedding: np.ndarray, top_k: int = 10):
    """
    Search index for nearest neighbours.

    Parameters
    ----------
    index     : faiss.Index
    embedding : np.ndarray of shape (1, D)
    top_k     : number of results

    Returns
    -------
    distances : np.ndarray (1, top_k) — cosine similarity scores
    indices   : np.ndarray (1, top_k) — gallery row positions
    """
    distances, indices = index.search(embedding.astype(np.float32), top_k)
    return distances, indices
