"""
src/api/app.py – FastAPI inference server.

Endpoints
---------
POST /predict/retriever   – pure Item2Vec retriever
POST /predict/ranker      – pure LACLRec ranker
POST /predict/combined    – two-stage retriever → ranker

Request body (JSON)
-------------------
{
    "sequence": [42, 17, 305, ...],   // list of item indices (int)
    "k": 10                           // optional, default 10
}

Response
--------
{
    "predictions": [<item_id>, ...]   // ranked list of k items
}

Start
-----
    uvicorn src.api.app:app --reload
    # or: python -m src.api.app
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.dataset import MLDataset
from src.retrieval.Item2Vec import Item2VecRetriever
from src.ranking.LACLRec import Encoder, Recommender
from src.models.retriever_ranker import RetrieverRankerRec
from src.models.ranker import build_ranker

from fastapi.responses import HTMLResponse

# ── paths ─────────────────────────────────────────────────────────────────────
PROCESSED_PT = os.environ.get("PROCESSED_PT", "datasets/processed.pt")
CKPT_DIR     = Path(os.environ.get("CKPT_DIR", "checkpoints"))
DEVICE       = os.environ.get("DEVICE", "cpu")
SEQ_LEN      = int(os.environ.get("SEQ_LEN", "50"))
EMBED_DIM    = int(os.environ.get("EMBED_DIM", "64"))
INSERT_LEN   = int(os.environ.get("INSERT_LEN", "3"))


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------


_dataset:    Optional[MLDataset]          = None
_retriever:  Optional[Item2VecRetriever]  = None
_ranker:     Optional[Recommender]        = None
_combined:   Optional[RetrieverRankerRec] = None


def _get_dataset() -> MLDataset:
    global _dataset
    if _dataset is None:
        _dataset = MLDataset.load_processed(PROCESSED_PT)
    return _dataset


def _get_retriever() -> Item2VecRetriever:
    global _retriever
    if _retriever is None:
        _retriever = Item2VecRetriever.load(str(CKPT_DIR / "retriever_a.pkl"))
        _retriever.load_faiss_index(str(CKPT_DIR / "retriever_a.faiss"))
    return _retriever


def _get_ranker() -> Recommender:
    global _ranker
    if _ranker is None:
        dataset = _get_dataset()
        encoder, _, recommender = build_ranker(
            n_items=dataset.num_items,
            seq_len=SEQ_LEN,
            embed_dim=EMBED_DIM,
            insert_len=INSERT_LEN,
        )
        encoder.load_state_dict(
            torch.load(CKPT_DIR / "ranker_a_encoder.pt", map_location=DEVICE)
        )
        recommender.load_state_dict(
            torch.load(CKPT_DIR / "ranker_a_recommender.pt", map_location=DEVICE)
        )
        encoder.to(DEVICE).eval()
        recommender.to(DEVICE).eval()
        _ranker = recommender
    return _ranker


def _get_combined() -> RetrieverRankerRec:
    global _combined
    if _combined is None:
        dataset = _get_dataset()
        retriever = _get_retriever()

        encoder, _, recommender = build_ranker(
            n_items=dataset.num_items,
            seq_len=SEQ_LEN,
            embed_dim=EMBED_DIM,
            insert_len=INSERT_LEN,
        )
        encoder.load_state_dict(
            torch.load(CKPT_DIR / "ranker_a_encoder.pt", map_location=DEVICE)
        )
        recommender.load_state_dict(
            torch.load(CKPT_DIR / "ranker_a_recommender.pt", map_location=DEVICE)
        )
        encoder.to(DEVICE).eval()
        recommender.to(DEVICE).eval()

        _combined = RetrieverRankerRec(
            retriever=retriever,
            recommender=recommender,
            device=DEVICE,
            n_candidates=1000,
        )
    return _combined


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(title="RecSys API", version="1.0")


class PredictRequest(BaseModel):
    sequence: list[int]   # original MovieLens movieId values
    k: int = 10


class PredictResponse(BaseModel):
    predictions: list[int]   # original MovieLens movieId values


def _map_input(movie_ids: list[int]) -> list[int]:
    """Map original movieIds → internal dataset indices. Unknown IDs are dropped."""
    item2idx = _get_dataset().item2idx
    mapped = [item2idx[mid] for mid in movie_ids if mid in item2idx]
    return mapped


def _map_output(internal_ids: list[int]) -> list[int]:
    """Map internal dataset indices → original movieIds. Unknown indices are dropped."""
    idx2item = _get_dataset().idx2item
    return [idx2item[iid] for iid in internal_ids if iid in idx2item]


def _pad_sequence(seq: list[int], seq_len: int) -> torch.Tensor:
    """Pad or truncate an already-mapped (internal index) sequence."""
    if len(seq) >= seq_len:
        seq = seq[-seq_len:]
    else:
        seq = [0] * (seq_len - len(seq)) + seq
    return torch.tensor(seq, dtype=torch.long)

@app.get("/", response_class=HTMLResponse)
def get_home():
    return """
    <html>
        <head>
            <title>Sequential RecSys API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }
                h1 {
                    color: #2c3e50;
                }
                .box {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                    max-width: 700px;
                }
                code {
                    background: #eee;
                    padding: 3px 6px;
                    border-radius: 4px;
                }
                a {
                    color: #3498db;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <div class="box">
                <h1>Sequential RecSys API</h1>
                <p>Welcome to the recommendation inference server.</p>

                <h3>Available Endpoints</h3>
                <ul>
                    <li><b>Retriever</b>: <code>/predict/retriever</code></li>
                    <li><b>Ranker</b>: <code>/predict/ranker</code></li>
                    <li><b>Retriever → Ranker</b>: <code>/predict/combined</code></li>
                </ul>

                <h3>API Documentation</h3>
                <p>
                    Interactive API docs are available at:<br>
                    <a href="/docs">/docs</a>
                </p>

                <h3>Health Check</h3>
                <p>
                    Server status endpoint:<br>
                    <code>/health</code>
                </p>

                <p style="margin-top:30px;color:gray;">
                    Sequential Recommendation System – FastAPI Inference Server
                </p>
            </div>
        </body>
    </html>
    """

@app.post("/predict/retriever", response_model=PredictResponse)
def predict_retriever(req: PredictRequest) -> PredictResponse:
    try:
        mapped = _map_input(req.sequence)
        if not mapped:
            raise ValueError(f"None of the provided movieIds are known: {req.sequence}")
        model = _get_retriever()
        internal_preds = model.predict_topk(mapped, k=req.k)
        return PredictResponse(predictions=_map_output(internal_preds))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/ranker", response_model=PredictResponse)
def predict_ranker(req: PredictRequest) -> PredictResponse:
    try:
        mapped = _map_input(req.sequence)
        if not mapped:
            raise ValueError(f"None of the provided movieIds are known: {req.sequence}")
        seq_t = _pad_sequence(mapped, SEQ_LEN).unsqueeze(0).to(DEVICE)
        model = _get_ranker()
        with torch.no_grad():
            logits = model(seq_t)[0]
        _, top_idx = torch.topk(logits, req.k)
        return PredictResponse(predictions=_map_output(top_idx.tolist()))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/combined", response_model=PredictResponse)
def predict_combined(req: PredictRequest) -> PredictResponse:
    try:
        mapped = _map_input(req.sequence)
        if not mapped:
            raise ValueError(f"None of the provided movieIds are known: {req.sequence}")
        seq_t = _pad_sequence(mapped, SEQ_LEN)
        model = _get_combined()
        internal_preds = model.predict_topk(seq_t, k=req.k)
        return PredictResponse(predictions=_map_output(internal_preds))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)