import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class ActionSequenceTokenizer:
    def __init__(self, max_length=50, stride=0, pad_token=0):
        self.max_length = max_length
        self.stride = stride
        self.pad_token = pad_token

    def encode(self, seq):
        seq = list(seq)
        sequences = []
        masks = []
        start = 0
        while start < len(seq):
            chunk = seq[start : start + self.max_length]
            mask = [1] * len(chunk)
            if len(chunk) < self.max_length:
                pad_size = self.max_length - len(chunk)
                chunk = chunk + [self.pad_token] * pad_size
                mask = mask + [0] * pad_size
            sequences.append(chunk)
            masks.append(mask)
            if len(seq) <= self.max_length:
                break
            start += self.max_length - self.stride
        return sequences, masks


class MLDataset(Dataset):
    """
    Builds (input_ids, label) pairs from a MovieLens-style ratings CSV.

    Each user sequence is split so that:
      - input_ids  = all items except the last
      - label      = the last item (next-item prediction target)

    Args:
        csv_path   : path to ratings.csv
        max_length : maximum sequence length (shorter seqs are left-padded with 0)
        stride     : overlap between consecutive windows (0 = no overlap)
    """

    def __init__(self, csv_path: str, max_length: int = 50, stride: int = 0):
        ratings_df = pd.read_csv(csv_path)

        # map movieId -> 1-based index (0 reserved for padding)
        item_ids = ratings_df["movieId"].unique()
        self.item2idx = {item: idx + 1 for idx, item in enumerate(sorted(item_ids))}
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        ratings_df["movieId"] = ratings_df["movieId"].map(self.item2idx)

        item_sequence = (
            ratings_df.sort_values(["userId", "timestamp"])
            .groupby("userId")["movieId"]
            .agg(list)
        )

        tokenizer = ActionSequenceTokenizer(max_length, stride)

        self.input_ids: list[list[int]] = []
        self.labels: list[int] = []
        self.sentences: list[list[str]] = []  # string tokens for Word2Vec

        for seq in item_sequence:
            if len(seq) < 2:
                continue
            seq_input = seq[:-1]
            label = seq[-1]
            ids_list, _ = tokenizer.encode(seq_input)
            for ids in ids_list:
                self.input_ids.append(ids)
                self.labels.append(label)
                self.sentences.append([str(i) for i in ids if i != 0])

        self.num_items = len(self.item2idx) + 1  # +1 for padding index 0
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_processed(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "input_ids": self.input_ids,
                "labels": self.labels,
                "sentences": self.sentences,
                "num_items": self.num_items,
                "item2idx": self.item2idx,
                "idx2item": self.idx2item,
            },
            path,
        )
        print(f"[MLDataset] saved to {path}")

    @classmethod
    def load_processed(cls, path: str) -> "MLDataset":
        obj = cls.__new__(cls)
        data = torch.load(path, weights_only=False)
        obj.input_ids = data["input_ids"]
        obj.labels = data["labels"]
        obj.sentences = data["sentences"]
        obj.num_items = data["num_items"]
        obj.item2idx = data.get("item2idx", {})
        obj.idx2item = data.get("idx2item", {})
        print(f"[MLDataset] loaded from {path}  ({len(obj)} samples)")
        return obj

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }