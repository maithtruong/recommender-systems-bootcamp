"""
RankerRec – pure LACLRec ranker (no retrieval stage).

Usage
-----
    from src.models.ranker import RankerRec, build_ranker

    encoder, augmenter, recommender = build_ranker(
        n_items=dataset.num_items,
        seq_len=50,
        embed_dim=64,
        insert_len=3,
    )
    # train with train_recommender(...)
    torch.save(recommender.state_dict(), "checkpoints/ranker_a.pt")
"""

from __future__ import annotations

import torch

from src.ranking.LACLRec import Augmenter, Encoder, Recommender


def build_ranker(
    n_items: int,
    seq_len: int = 50,
    embed_dim: int = 64,
    insert_len: int = 3,
) -> tuple[Encoder, Augmenter, Recommender]:
    """Construct and return (encoder, augmenter, recommender)."""
    encoder = Encoder(n_items=n_items, seq_len=seq_len, embed_dim=embed_dim)
    augmenter = Augmenter(
        encoder=encoder,
        n_items=n_items,
        max_len=seq_len,
        insert_len=insert_len,
    )
    recommender = Recommender(encoder=encoder)
    return encoder, augmenter, recommender


# Alias so imports from src.models stay consistent
RankerRec = Recommender