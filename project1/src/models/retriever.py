"""
RetrieverRec – pure Item2Vec retrieval model.

Usage
-----
    from src.models.retriever import RetrieverRec

    model = RetrieverRec(sentences=dataset.sentences)
    model.save("checkpoints/retriever_a.pkl")

    model = RetrieverRec.load("checkpoints/retriever_a.pkl")
    top_k = model.predict_topk(seq, k=10)
"""

from __future__ import annotations

from src.retrieval.Item2Vec import Item2VecRetriever


class RetrieverRec(Item2VecRetriever):
    """
    Thin wrapper so downstream code can import from `src.models`
    without knowing about the retrieval sub-package.
    """

    def save(self, path: str) -> None:
        """Persist model weights to *path* (gensim native format)."""
        self.save(path)                     # gensim Word2Vec.save

    @classmethod
    def load(cls, path: str) -> "RetrieverRec":
        """Load a previously saved RetrieverRec."""
        instance = cls.__new__(cls)
        loaded = Item2VecRetriever.load(path)
        instance.__dict__.update(loaded.__dict__)
        return instance