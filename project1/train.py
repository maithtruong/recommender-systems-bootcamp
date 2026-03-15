"""
train.py – Unified training entry point.

Environment variables (set in .env or shell):
    RATINGS_CSV   path to the MovieLens ratings.csv
    PROCESSED_PT  path to save/load the processed dataset  (default: datasets/processed.pt)
    CKPT_DIR      directory for model checkpoints          (default: checkpoints/)

Usage examples
--------------
    # 1. Process raw CSV and save to disk (no training)
    python train.py --mode process

    # 2. Train retriever A  (Item2Vec on all data)
    python train.py --mode retriever_a

    # 3. Train ranker A  (LACLRec on all data)
    python train.py --mode ranker_a

Combined model B requires no training — it composes retriever_a + ranker_a at inference time.
See src/models/retriever_ranker.py and the evaluation notebook.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── project imports ───────────────────────────────────────────────────────────
from src.data.dataset import MLDataset
from src.retrieval.Item2Vec import Item2VecRetriever
from src.ranking.LACLRec import train_recommender
from src.models.ranker import build_ranker

# ── paths from environment ────────────────────────────────────────────────────
RATINGS_CSV  = os.environ.get("RATINGS_CSV",  "datasets/ml-latest-small/ratings.csv")
PROCESSED_PT = os.environ.get("PROCESSED_PT", "datasets/processed.pt")
CKPT_DIR     = Path(os.environ.get("CKPT_DIR", "checkpoints"))
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── hyper-parameters (override via env if needed) ─────────────────────────────
SEQ_LEN    = int(os.environ.get("SEQ_LEN",    "50"))
EMBED_DIM  = int(os.environ.get("EMBED_DIM",  "64"))
INSERT_LEN = int(os.environ.get("INSERT_LEN", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
EPOCHS     = int(os.environ.get("EPOCHS",     "5"))
LR         = float(os.environ.get("LR",       "1e-3"))
DEVICE     = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_dataset() -> MLDataset:
    if Path(PROCESSED_PT).exists():
        print(f"[train] loading processed dataset from {PROCESSED_PT}")
        return MLDataset.load_processed(PROCESSED_PT)
    print(f"[train] processing raw CSV {RATINGS_CSV}")
    dataset = MLDataset(RATINGS_CSV, max_length=SEQ_LEN)
    dataset.save_processed(PROCESSED_PT)
    return dataset


def make_loader(dataset: MLDataset, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------


def train_retriever_a(dataset: MLDataset) -> None:
    """Train Item2Vec on *all* user sentences, then build FAISS index."""
    print("[train] --- Retriever A (Item2Vec, all data) ---")
    model = Item2VecRetriever(
        sentences=dataset.sentences,
        vector_size=EMBED_DIM,
        epochs=EPOCHS,
    )
    save_path = str(CKPT_DIR / "retriever_a.pkl")
    model.save(save_path)
    print(f"[train] retriever_a saved → {save_path}")

    index_path = str(CKPT_DIR / "retriever_a.faiss")
    model.build_faiss_index(index_path)


def train_ranker_a(dataset: MLDataset) -> None:
    """Train LACLRec on *all* data (pure ranker A)."""
    print("[train] --- Ranker A (LACLRec, all data) ---")
    encoder, augmenter, recommender = build_ranker(
        n_items=dataset.num_items,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        insert_len=INSERT_LEN,
    )
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(recommender.parameters()),
        lr=LR,
    )
    loader = make_loader(dataset)
    train_recommender(encoder, augmenter, recommender, loader, optimizer, epochs=EPOCHS, device=DEVICE)

    torch.save(encoder.state_dict(),      CKPT_DIR / "ranker_a_encoder.pt")
    torch.save(recommender.state_dict(),  CKPT_DIR / "ranker_a_recommender.pt")
    print(f"[train] ranker_a saved → {CKPT_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RecSys training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Combined model B needs no training — it composes retriever_a + ranker_a\n"
            "at inference time via src/models/retriever_ranker.py."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["process", "retriever_a", "ranker_a"],
        required=True,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "process":
        dataset = MLDataset(RATINGS_CSV, max_length=SEQ_LEN)
        dataset.save_processed(PROCESSED_PT)
        return

    dataset = load_dataset()

    if args.mode == "retriever_a":
        train_retriever_a(dataset)
    elif args.mode == "ranker_a":
        train_ranker_a(dataset)


if __name__ == "__main__":
    main()