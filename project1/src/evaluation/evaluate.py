"""
Evaluation helpers.

Metrics  : Recall@K, NDCG@K
Evaluators:
    evaluate_Item2Vec        – pure retriever
    evaluate_LACLRec         – pure ranker
    evaluate_RetrieverRanker – two-stage pipeline
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def recall_at_k(
    preds: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    """
    Args:
        preds  : (B, ≥k) predicted item indices
        targets: (B,)    ground-truth item indices
        k      : cutoff

    Returns:
        Mean Recall@K over the batch (float in [0, 1])
    """
    if preds.ndim == 1:
        preds = preds.unsqueeze(0)
        targets = targets.unsqueeze(0) if targets.ndim == 0 else targets

    preds = preds[:, :k]
    hits = (preds == targets.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


def ndcg_at_k(
    preds: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    """
    Args:
        preds  : (B, ≥k) predicted item indices
        targets: (B,)    ground-truth item indices
        k      : cutoff

    Returns:
        Mean NDCG@K over the batch (float in [0, 1])
    """
    if preds.ndim == 1:
        preds = preds.unsqueeze(0)
        targets = targets.unsqueeze(0) if targets.ndim == 0 else targets

    preds = preds[:, :k]
    batch_size = preds.size(0)
    score = 0.0
    for i in range(batch_size):
        match = (preds[i] == targets[i]).nonzero(as_tuple=True)
        if len(match[0]) > 0:
            rank = match[0][0].item() + 1          # 1-based rank
            score += 1.0 / math.log2(rank + 1)
    return score / batch_size


# ---------------------------------------------------------------------------
# Per-model evaluators
# ---------------------------------------------------------------------------


def evaluate_Item2Vec(
    model,                          # Item2VecRetriever
    loader,                         # DataLoader over MLDataset
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a pure Item2Vec retriever."""
    recall_total = ndcg_total = steps = 0

    for batch in loader:
        seqs = batch["input_ids"]       # (B, T)
        labels = batch["labels"]        # (B,)

        batch_preds = []
        for seq in seqs:
            preds = model.predict_topk(seq.tolist(), k=k)
            if len(preds) < k:
                preds += [0] * (k - len(preds))
            batch_preds.append(preds)

        preds_t = torch.tensor(batch_preds)
        recall_total += recall_at_k(preds_t, labels, k)
        ndcg_total   += ndcg_at_k(preds_t, labels, k)
        steps += 1

    return {
        f"Recall@{k}": recall_total / max(steps, 1),
        f"NDCG@{k}":   ndcg_total   / max(steps, 1),
    }


def evaluate_LACLRec(
    model,                          # LACLRec Recommender
    loader,                         # DataLoader over MLDataset
    device: str,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a pure LACLRec ranker."""
    model.eval()
    recall_total = ndcg_total = steps = 0

    with torch.no_grad():
        for batch in loader:
            seq    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(seq)                         # (B, n_items)
            _, preds = torch.topk(logits, k, dim=1)    # (B, k)

            recall_total += recall_at_k(preds, labels, k)
            ndcg_total   += ndcg_at_k(preds, labels, k)
            steps += 1

    return {
        f"Recall@{k}": recall_total / max(steps, 1),
        f"NDCG@{k}":   ndcg_total   / max(steps, 1),
    }


def evaluate_RetrieverRanker(
    model,                          # RetrieverRankerRec
    loader,                         # DataLoader over MLDataset
    k: int = 10,
) -> dict[str, float]:
    """Evaluate the two-stage retrieval→ranking pipeline."""
    recall_total = ndcg_total = steps = 0

    for batch in loader:
        seqs   = batch["input_ids"]     # (B, T)
        labels = batch["labels"]        # (B,)

        preds_t = model.evaluate_batch(seqs, k=k)   # (B, k)

        recall_total += recall_at_k(preds_t, labels, k)
        ndcg_total   += ndcg_at_k(preds_t, labels, k)
        steps += 1

    return {
        f"Recall@{k}": recall_total / max(steps, 1),
        f"NDCG@{k}":   ndcg_total   / max(steps, 1),
    }