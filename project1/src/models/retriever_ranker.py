"""
RetrieverRankerRec – two-stage pipeline (Combined model B).

No separate training is required.  The pipeline composes:
  - a trained Item2VecRetriever  (retriever_a)
  - a trained LACLRec Recommender (ranker_a)

Stage 1 (retrieve) : Item2Vec retrieves `n_candidates` items from the full catalogue
Stage 2 (rank)     : LACLRec scores those candidates and returns the top-k

Usage
-----
    from src.models.retriever_ranker import RetrieverRankerRec, load_combined

    model = load_combined(dataset, ckpt_dir="checkpoints/", device="cpu")
    top_k = model.predict_topk(seq_tensor, k=10)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from src.retrieval.Item2Vec import Item2VecRetriever
from src.ranking.LACLRec import Recommender
from src.models.ranker import build_ranker


class RetrieverRankerRec:
    """
    Two-stage retrieval → ranking pipeline.

    Parameters
    ----------
    retriever     : trained Item2VecRetriever
    recommender   : trained LACLRec Recommender
    device        : torch device string
    n_candidates  : number of candidates retrieved before re-ranking
    """

    def __init__(
        self,
        retriever: Item2VecRetriever,
        recommender: Recommender,
        device: str = "cpu",
        n_candidates: int = 1000,
    ):
        self.retriever = retriever
        self.recommender = recommender
        self.device = device
        self.n_candidates = n_candidates
        self.recommender.eval()

    # ------------------------------------------------------------------

    def predict_topk(self, seq: torch.Tensor, k: int = 10) -> list[int]:
        """
        Args:
            seq : (T,) padded item-index tensor for ONE user
            k   : number of final items to return

        Returns:
            List of k item indices (ranked best-first)
        """
        # Stage 1 – retrieve candidates
        seq_list = seq.tolist()
        candidates = self.retriever.predict_topk(seq_list, k=self.n_candidates)
        if not candidates:
            return []

        # Stage 2 – score candidates with the recommender
        with torch.no_grad():
            seq_batch = seq.unsqueeze(0).to(self.device)       # (1, T)
            logits = self.recommender(seq_batch)[0]             # (n_items,)

        # Keep only logits for retrieved candidates, then rank
        candidate_tensor = torch.tensor(candidates, device=self.device)
        candidate_logits = logits[candidate_tensor]
        ranked_idx = torch.argsort(candidate_logits, descending=True)[:k]
        return [candidates[i] for i in ranked_idx.tolist()]

    # ------------------------------------------------------------------
    # Batch evaluation helper (mirrors evaluate_LACLRec signature)
    # ------------------------------------------------------------------

    def evaluate_batch(self, seq_batch: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Args:
            seq_batch : (B, T)
            k         : top-k to return

        Returns:
            preds: (B, k) long tensor
        """
        preds = []
        for seq in seq_batch:
            top = self.predict_topk(seq, k=k)
            if len(top) < k:
                top = top + [0] * (k - len(top))
            preds.append(top)
        return torch.tensor(preds, dtype=torch.long)


# ---------------------------------------------------------------------------
# Convenience loader — no retraining needed
# ---------------------------------------------------------------------------


def load_combined(
    dataset,
    ckpt_dir: str = "checkpoints/",
    device: str = "cpu",
    n_candidates: int = 1000,
    seq_len: int = 50,
    embed_dim: int = 64,
    insert_len: int = 3,
) -> "RetrieverRankerRec":
    """
    Build Combined model B by loading retriever_a + ranker_a from disk.

    No separate training step required — just call this after training both
    individual models.

    Args:
        dataset      : MLDataset (needed for num_items)
        ckpt_dir     : directory containing retriever_a.pkl and ranker_a_*.pt
        device       : torch device string
        n_candidates : how many candidates retriever passes to ranker
        seq_len      : must match training config
        embed_dim    : must match training config
        insert_len   : must match training config

    Returns:
        Ready-to-use RetrieverRankerRec instance
    """
    ckpt = Path(ckpt_dir)

    retriever = Item2VecRetriever.load(str(ckpt / "retriever_a.pkl"))
    retriever.load_faiss_index(str(ckpt / "retriever_a.faiss"))

    encoder, _, recommender = build_ranker(
        n_items=dataset.num_items,
        seq_len=seq_len,
        embed_dim=embed_dim,
        insert_len=insert_len,
    )
    encoder.load_state_dict(
        torch.load(ckpt / "ranker_a_encoder.pt", map_location=device)
    )
    recommender.load_state_dict(
        torch.load(ckpt / "ranker_a_recommender.pt", map_location=device)
    )
    encoder.to(device).eval()
    recommender.to(device).eval()

    return RetrieverRankerRec(
        retriever=retriever,
        recommender=recommender,
        device=device,
        n_candidates=n_candidates,
    )