# RecSys Project — Sequential Recommendation System

A complete end-to-end recommendation system covering the full ML and engineering pipeline:
data preprocessing, retrieval with ANN search, transformer-based ranking, offline evaluation,
and REST API serving.

Built as part of the **RecSys Bootcamp at Cole.vn**, hosted by Dr. Le Thien Hoa.

---

## About

This project implements a production-style sequential recommendation pipeline on the
MovieLens dataset. The system follows the two-stage retrieve-then-rank architecture
standard in industry, where a fast retrieval model narrows the full item catalogue
to a candidate set, and a precise ranking model reorders those candidates for the
final recommendation.

**Results (MovieLens-small, @K=10):**

| Model | Recall@10 | NDCG@10 |
|-------|-----------|---------|
| Retriever A (Item2Vec + FAISS) | 0.0139 | 0.0064 |
| Ranker A (LACLRec) | 0.1125 | 0.0557 |
| Combined B (Retriever → Ranker) | 0.0769 | 0.0360 |

The standalone LACLRec ranker achieved the best performance.

---

## Objective

The goal is to simulate a real-world production recommendation pipeline by implementing
every stage end-to-end:

- **Data preprocessing** — convert raw interaction logs into padded user sequences
- **Retrieval model** — Item2Vec (skip-gram) trained on interaction sequences
- **Approximate nearest neighbour search** — FAISS index over item embeddings
- **Ranking model** — LACLRec, a transformer trained with self-supervised contrastive learning
- **REST API serving** — FastAPI server with endpoints for each model and the combined pipeline
- **Offline evaluation** — Recall@K and NDCG@K with ablation across all three configurations

The pipeline covers all mandatory requirements from the bootcamp specification.
The optional LLM reranker stage is not implemented in this version.

```
User sequence  →  Retriever (Item2Vec + FAISS)
                       ↓ top-1000 candidates
               Ranker (LACLRec)
                       ↓ top-k results
```

---

## Setup

### Requirements
 
```
torch
gensim
faiss-cpu          # or faiss-gpu for CUDA
fastapi
uvicorn
pandas
numpy
jupyter
notebook
ipykernel
```
 
The project uses **uv** for dependency management. To activate the environment:
 
```bash
source .venv/bin/activate
```
 
All dependencies are declared in the project.

### Environment variables

| Variable       | Default                                  | Description                     |
|----------------|------------------------------------------|---------------------------------|
| `RATINGS_CSV`  | `datasets/ml-latest-small/ratings.csv`   | Path to raw MovieLens CSV       |
| `PROCESSED_PT` | `datasets/processed.pt`                  | Serialised dataset              |
| `CKPT_DIR`     | `checkpoints/`                           | Model checkpoint directory      |
| `SEQ_LEN`      | `50`                                     | Max sequence length             |
| `EMBED_DIM`    | `64`                                     | Embedding / hidden dimension    |
| `INSERT_LEN`   | `3`                                      | Max items inserted by augmenter |
| `BATCH_SIZE`   | `64`                                     | Training batch size             |
| `EPOCHS`       | `5`                                      | Training epochs                 |
| `LR`           | `1e-3`                                   | Learning rate                   |
| `DEVICE`       | `cuda` if available, else `cpu`          | PyTorch device                  |

### Dataset

Download the MovieLens dataset and place it at:

```
datasets/ml-latest-small/ratings.csv
```

---

## Project Structure

```
recsys-project/
│
├── src/
│   ├── data/
│   │   └── dataset.py              # MLDataset + ActionSequenceTokenizer
│   ├── retrieval/
│   │   └── Item2Vec.py             # Item2VecRetriever — Word2Vec + FAISS
│   ├── ranking/
│   │   └── LACLRec.py              # Encoder, Augmenter, Recommender, training loop
│   ├── models/
│   │   ├── retriever.py            # RetrieverRec alias
│   │   ├── ranker.py               # build_ranker() factory
│   │   └── retriever_ranker.py     # RetrieverRankerRec + load_combined()
│   ├── evaluation/
│   │   └── evaluate.py             # recall_at_k, ndcg_at_k, per-model evaluators
│   └── api/
│       └── app.py                  # FastAPI inference server
│
├── datasets/                       # Raw CSV + processed .pt files
├── checkpoints/                    # Saved model weights + FAISS index
├── notebooks/
│   └── evaluation_report.ipynb    # End-to-end training + evaluation
├── train.py                        # CLI training entry point
└── README.md
```

---

## Models

### Retriever A — Item2Vec

**File:** `src/retrieval/Item2Vec.py`

A Word2Vec model trained on user interaction sequences. Each item is treated as a
"word" and each user session as a "sentence". At inference time, the mean-pooled
embedding of the input sequence is used to query a FAISS index for fast approximate
nearest-neighbour retrieval.

| Property | Value |
|----------|-------|
| Architecture | Word2Vec (Skip-gram) |
| Index | FAISS `IndexFlatIP` (cosine similarity) |
| Output | Top-k item indices |
| Checkpoints | `retriever_a.pkl`, `retriever_a.faiss`, `retriever_a.items.pkl` |

---

### Ranker A — LACLRec

**File:** `src/ranking/LACLRec.py`

A transformer-based sequential recommender trained with a self-supervised contrastive
learning objective. The model produces two augmented views of each sequence (one
learned, one random) and aligns their representations using NT-Xent loss, alongside
standard next-item cross-entropy loss.

**Architecture:**

```
Input sequence
      ↓
Token embedding + Positional embedding
      ↓
Transformer Encoder (4 layers, 8 heads)   ← shared Encoder
      ↓
Last hidden state h_T
      ↓
Dot product with item embedding matrix
      ↓
Logits over all items → Top-k
```

**Augmentation (training only):**

```
For each item in sequence, apply one of:
  keep   → unchanged
  delete → drop item
  insert → ReverseGenerator inserts up to insert_len items before it
```

| Property | Value |
|----------|-------|
| Architecture | Transformer encoder (4L × 8H) |
| Loss | Cross-entropy (rec) + NT-Xent (SSL contrastive) |
| Output | Logits over all items |
| Checkpoints | `ranker_a_encoder.pt`, `ranker_a_recommender.pt` |

---

### Combined B — RetrieverRankerRec

**File:** `src/models/retriever_ranker.py`

A two-stage pipeline that composes Retriever A and Ranker A. No additional training
is required — the two models are loaded from their existing checkpoints and wired
together at inference time.

```
Input sequence
      ↓
Retriever A  →  FAISS query  →  top-1000 candidates
                                        ↓
                              Ranker A scores candidates
                                        ↓
                                    Top-k items
```

| Property | Value |
|----------|-------|
| Stage 1 | Item2Vec + FAISS (1000 candidates) |
| Stage 2 | LACLRec re-ranks candidates by logit score |
| Extra training | None |

---

## Training

```bash
# Step 1 — process raw CSV into tokenised sequences
python train.py --mode process

# Step 2 — train retriever (also builds FAISS index automatically)
python train.py --mode retriever_a

# Step 3 — train ranker
python train.py --mode ranker_a
```

Checkpoints written after training:

```
checkpoints/
  retriever_a.pkl          ← Word2Vec weights
  retriever_a.faiss        ← FAISS index
  retriever_a.items.pkl    ← FAISS position → item ID map
  ranker_a_encoder.pt      ← Transformer encoder weights
  ranker_a_recommender.pt  ← Recommender head weights
```

### Training on Colab (no local GPU)
 
> Don't have a GPU? You can run training on Google Colab with a free T4/A100.
>
> 1. Upload the `project1/` repo to your Google Drive
> 2. Clone or download the Colab-compatible `evaluation_report.ipynb`: **[train.py for Colab](https://drive.google.com/file/d/1USWpz_MQx7rLLKPdFlSTdb94Mod9g4eV/view?usp=sharing)**
> 3. Replace the existing `evaluation_report.ipynb` in your uploaded repo with this file
> 4. Open `notebooks/evaluation_report.ipynb` in Colab and run all cells

---

## Evaluation

Run the full training + evaluation pipeline in one notebook:

```bash
jupyter notebook notebooks/evaluation_report.ipynb
```

Run all cells top to bottom. The notebook trains all models, then evaluates each
at Recall@10 and NDCG@10 and displays a summary table.

### Metrics

**Recall@K** — fraction of users for whom the ground-truth item appears in the top-K
predictions.

**NDCG@K** — normalised discounted cumulative gain; rewards correct predictions that
appear higher in the ranking.

### Expected output (example)

| Model | Recall@10 | NDCG@10 |
|-------|-----------|---------|
| Retriever A (Item2Vec + FAISS) | 0.xxxx | 0.xxxx |
| Ranker A (LACLRec) | 0.xxxx | 0.xxxx |
| Combined B (Retriever → Ranker) | 0.xxxx | 0.xxxx |

---

## Inference API

Start the server:

```bash
uvicorn src.api.app:app --reload
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict/retriever` | Item2Vec + FAISS top-k |
| `POST` | `/predict/ranker` | LACLRec top-k over all items |
| `POST` | `/predict/combined` | Two-stage retrieval → ranking |
| `GET`  | `/health` | Liveness check |

> **Note:** All `sequence` values are **original MovieLens movieIds**. The API maps them to internal indices automatically and returns predictions as original movieIds.

---

### Option 1 — Interactive UI (easiest)

Open the auto-generated FastAPI docs in your browser:

```
http://127.0.0.1:8000/docs
```

Steps:
1. Scroll to `POST /predict/retriever` (or `/ranker`, `/combined`)
2. Click **Try it out**
3. Enter your request body and click **Execute**:

```json
{
  "sequence": [12, 45, 78, 23],
  "k": 10
}
```

Example response:

```json
{
  "predictions": [88, 102, 31, 77, 56, 90, 11, 5, 67, 201]
}
```

---

### Option 2 — curl (terminal)

```bash
curl -X POST "http://127.0.0.1:8000/predict/retriever" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [12, 45, 78, 23],
    "k": 10
  }'
```

Replace `/predict/retriever` with `/predict/ranker` or `/predict/combined` to use a different model.

---

### Option 3 — Python

```python
import requests

url = "http://127.0.0.1:8000/predict/combined"

data = {
    "sequence": [12, 45, 78, 23],
    "k": 10
}

r = requests.post(url, json=data)
print(r.json())
# {'predictions': [88, 102, 31, ...]}
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                               │
│                                                                     │
│  ┌─────────────────────┐       ┌─────────────────────────────────┐  │
│  │   Data Preparation  │       │           Training              │  │
│  │                     │       │                                 │  │
│  │  ratings.csv        │       │  ┌─────────────┐                │  │
│  │       ↓             │       │  │  Retriever  │                │  │
│  │  Tokenize sequences │       │  │  (Item2Vec) │                │  │
│  │       ↓             │       │  └──────┬──────┘                │  │
│  │  Save processed.pt  │──────▶│         │ embeddings            │  │
│  │                     │       │         ↓                       │  │
│  └─────────────────────┘       │  ┌─────────────┐                │  │
│                                │  │ FAISS Index │                │  │
│                                │  └─────────────┘                │  │
│                                │                                 │  │
│                                │  ┌─────────────┐                │  │
│                                │  │   Ranker    │                │  │
│                                │  │  (LACLRec)  │                │  │
│                                │  └──────┬──────┘                │  │
│                                │         │ weights               │  │
│                                │         ↓                       │  │
│                                │  ┌─────────────┐                │  │
│                                │  │ encoder.pt  │                │  │
│                                │  │ recomm.pt   │                │  │
│                                │  └─────────────┘                │  │
│                                └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                              │
│                                                                     │
│   API Request (sequence)                                            │
│          │                                                          │
│          ├──────────────────────┬──────────────────────────────┐    │
│          ▼                      ▼                              │    │
│  ┌───────────────────┐  ┌───────────────────┐                  │    │
│  │  Retriever        │  │  Ranker           │  Combined B:     │    │
│  │                   │  │                   │                  │    │
│  │  Embed sequence   │  │  Encode sequence  │  Retriever →     │    │
│  │        ↓          │  │        ↓          │  FAISS query →   │    │
│  │  Query FAISS      │  │  Compute logits   │  Ranker scores   │    │
│  │        ↓          │  │        ↓          │  candidates →    │    │
│  │  Top-k items      │  │  Top-k items      │  Top-k items     │    │
│  └───────────────────┘  └───────────────────┘                  │    │
│          │                      │                              │    │
│          └──────────────────────┴──────────────────────────────┘    │
│                                 ↓                                   │
│                          API Response                               │
└─────────────────────────────────────────────────────────────────────┘
```