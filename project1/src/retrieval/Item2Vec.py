from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from gensim.models import Word2Vec


class Item2VecRetriever(Word2Vec):
    """
    Word2Vec-based item retriever with FAISS index for fast ANN search.

    After training, call `build_faiss_index()` once to build and save the
    index.  At inference time, load the saved index with `load_faiss_index()`
    and use `predict_topk()` which queries FAISS instead of gensim's
    brute-force search.
    """

    def __init__(
        self,
        sentences=None,
        vector_size: int = 128,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        epochs: int = 10,
        **kwargs,
    ):
        super().__init__(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            **kwargs,
        )
        self._faiss_index = None
        self._faiss_items: list[int] = []   # position i → item int id

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def get_sequence_embedding(self, seq: list) -> np.ndarray | None:
        """Mean-pool the Word2Vec vectors of the items in *seq*."""
        vectors = []
        for item in seq:
            key = str(item)
            if key != "0" and key in self.wv:
                vectors.append(self.wv[key])
        if not vectors:
            return None
        return np.mean(vectors, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # FAISS index: build and save
    # ------------------------------------------------------------------

    def build_faiss_index(self, index_path: str) -> None:
        """
        Build a flat L2 FAISS index over all trained item embeddings and
        save it to *index_path*.  Also saves a companion item-list file
        so integer item IDs can be recovered from FAISS positions.

        Call once after Word2Vec training; not needed again unless
        the model is retrained.
        """
        vocab = list(self.wv.key_to_index.keys())
        vectors = np.array([self.wv[w] for w in vocab], dtype=np.float32)
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)   # inner-product on normalised vecs = cosine
        index.add(vectors)

        faiss.write_index(index, str(index_path))

        items_path = Path(index_path).with_suffix(".items.pkl")
        with open(items_path, "wb") as f:
            pickle.dump([int(w) for w in vocab], f)

        self._faiss_index = index
        self._faiss_items = [int(w) for w in vocab]
        print(f"[Item2Vec] FAISS index saved → {index_path}  ({len(vocab)} items)")

    # ------------------------------------------------------------------
    # FAISS index: load
    # ------------------------------------------------------------------

    def load_faiss_index(self, index_path: str) -> None:
        """Load a previously saved FAISS index + companion item list."""
        self._faiss_index = faiss.read_index(str(index_path))

        items_path = Path(index_path).with_suffix(".items.pkl")
        with open(items_path, "rb") as f:
            self._faiss_items = pickle.load(f)

        print(f"[Item2Vec] FAISS index loaded ← {index_path}  ({len(self._faiss_items)} items)")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def predict_topk(self, seq: list, k: int = 10) -> list[int]:
        """
        Return top-k item indices most similar to the sequence.

        Uses FAISS if an index is loaded, otherwise falls back to
        gensim brute-force (useful during training before index is built).
        """
        seq_emb = self.get_sequence_embedding(seq)
        if seq_emb is None:
            return []

        if self._faiss_index is not None:
            vec = seq_emb.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(vec)
            _, indices = self._faiss_index.search(vec, k)
            return [self._faiss_items[i] for i in indices[0] if i < len(self._faiss_items)]

        # fallback: gensim brute-force
        neighbors = self.wv.similar_by_vector(seq_emb, topn=k)
        return [int(item) for item, _ in neighbors]