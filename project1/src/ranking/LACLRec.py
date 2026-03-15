"""
LACLRec – sequence encoder + SSL augmenter + recommender.

Paper components
----------------
Encoder          : shared transformer that maps item sequences → hidden states
ReverseGenerator : auto-regressively inserts items into an augmented sequence
SSLAugmenter     : keep / delete / insert operations driven by the encoder
RandomAugmenter  : cheap random baseline for the second augmented view
Augmenter        : orchestrates SSL + random views (used at training time)
Recommender      : thin head on top of the shared encoder
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder (shared between augmenter and recommender)
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """
    Maps an item-index sequence to contextual hidden representations.

    Architecture
    ------------
    token embedding  +  positional embedding  →  L-layer Transformer
    """

    def __init__(self, n_items: int, seq_len: int, embed_dim: int):
        super().__init__()
        self.token_embedder = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.pos_embedder = nn.Embedding(seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: (B, T) long tensor of item indices

        Returns:
            H: (B, T, embed_dim) contextual representations
        """
        token = self.token_embedder(seq)
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        pos = self.pos_embedder(positions)
        H = self.transformer(token + pos)
        return H


# ---------------------------------------------------------------------------
# Reverse generator (insert sub-sequence)
# ---------------------------------------------------------------------------


class ReverseGenerator(nn.Module):
    """
    Given the hidden state h_t of the *anchor* item at position t,
    auto-regressively generates up to `insert_len` items to insert
    before that position.
    """

    def __init__(
        self,
        item_embedding: nn.Embedding,
        max_len: int,
        insert_len: int,
    ):
        super().__init__()
        self.item_embedding = item_embedding
        self.insert_len = insert_len
        embed_dim = item_embedding.embedding_dim

        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, ht: torch.Tensor) -> list[int]:
        """
        Args:
            ht: (embed_dim,) hidden state for the anchor item

        Returns:
            List of generated item indices (length == insert_len)
        """
        gen_seq: list[int] = []

        for _ in range(self.insert_len):
            if len(gen_seq) == 0:
                stack = ht.unsqueeze(0)                         # (1, E)
            else:
                gen_tensor = torch.tensor(gen_seq, device=ht.device)
                gen_embed = self.item_embedding(gen_tensor)     # (n, E)
                stack = torch.cat([ht.unsqueeze(0), gen_embed], dim=0)

            pos = torch.arange(stack.size(0), device=ht.device)
            stack = stack + self.pos_embedding(pos)
            H = self.transformer(stack.unsqueeze(0))            # (1, n, E)

            logits = torch.matmul(H[:, -1, :], self.item_embedding.weight.T)
            next_item = torch.argmax(logits).item()
            gen_seq.append(int(next_item))

        return gen_seq


# ---------------------------------------------------------------------------
# SSL augmenter
# ---------------------------------------------------------------------------


class SSLAugmenter(nn.Module):
    """
    Selects keep / delete / insert for each position, guided by the encoder.
    """

    def __init__(
        self,
        item_embedding: nn.Embedding,
        max_len: int,
        insert_len: int,
    ):
        super().__init__()
        embed_dim = item_embedding.embedding_dim
        self.operation_selector = nn.Linear(embed_dim, 3)
        self.reverse_generator = ReverseGenerator(item_embedding, max_len, insert_len)

    def forward(self, seq: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq : (T,) 1-D item-index tensor (single sequence, no batch dim)
            H   : (T, embed_dim) encoder output for that sequence

        Returns:
            aug_seq: 1-D long tensor (variable length)
        """
        aug: list[int] = []
        for t in range(seq.size(0)):
            ht = H[t]
            probs = torch.softmax(self.operation_selector(ht), dim=-1)
            op = torch.multinomial(probs, 1).item()

            if op == 0:                             # keep
                aug.append(seq[t].item())
            elif op == 1:                           # delete
                pass
            else:                                   # insert then keep
                aug.extend(self.reverse_generator(ht))
                aug.append(seq[t].item())

        return torch.tensor(aug, dtype=torch.long, device=seq.device)


# ---------------------------------------------------------------------------
# Random augmenter
# ---------------------------------------------------------------------------


class RandomAugmenter(nn.Module):
    """Cheap stochastic baseline augmenter (no learned parameters)."""

    def __init__(self, n_items: int, insert_len: int):
        super().__init__()
        self.n_items = n_items
        self.insert_len = insert_len

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        aug: list[int] = []
        for item in seq:
            op = random.choice(["keep", "delete", "insert"])
            if op == "keep":
                aug.append(item.item())
            elif op == "delete":
                pass
            else:
                for _ in range(self.insert_len):
                    aug.append(random.randint(1, self.n_items - 1))
                aug.append(item.item())
        return torch.tensor(aug, dtype=torch.long, device=seq.device)


# ---------------------------------------------------------------------------
# Augmenter orchestrator
# ---------------------------------------------------------------------------


class Augmenter:
    """
    Produces two augmented views of each sequence:
      aug1 – SSL-guided (learned operations)
      aug2 – random

    Both are normalised to `max_len` (truncated or left-padded).
    """

    def __init__(
        self,
        encoder: Encoder,
        n_items: int,
        max_len: int,
        insert_len: int,
    ):
        self.encoder = encoder
        self.ssl_aug = SSLAugmenter(encoder.token_embedder, max_len, insert_len)
        self.rand_aug = RandomAugmenter(n_items, insert_len)
        self.max_len = max_len

    def _normalize(self, seq: torch.Tensor) -> torch.Tensor:
        if len(seq) > self.max_len:
            seq = seq[-self.max_len :]
        if len(seq) < self.max_len:
            pad = torch.zeros(self.max_len - len(seq), dtype=torch.long, device=seq.device)
            seq = torch.cat([pad, seq])
        return seq

    @torch.no_grad()
    def __call__(
        self, seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq: (max_len,) padded sequence for a single user

        Returns:
            (aug1, aug2) – both normalised to (max_len,)
        """
        H = self.encoder(seq.unsqueeze(0))[0]      # (T, E)
        aug1 = self.ssl_aug(seq, H)
        aug2 = self.rand_aug(seq)
        return self._normalize(aug1), self._normalize(aug2)


# ---------------------------------------------------------------------------
# Recommender head
# ---------------------------------------------------------------------------


class Recommender(nn.Module):
    """Next-item prediction head on top of the shared Encoder."""

    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: (B, T) padded item-index tensor

        Returns:
            logits: (B, n_items)
        """
        H = self.encoder(seq)
        h_last = H[:, -1, :]
        logits = torch.matmul(h_last, self.encoder.token_embedder.weight.T)
        return logits


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent (InfoNCE) contrastive loss for self-supervised pre-training."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_recommender(
    encoder: Encoder,
    augmenter: Augmenter,
    recommender: Recommender,
    loader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 2,
    device: str = "cpu",
) -> None:
    encoder.to(device)
    recommender.to(device)
    encoder.train()
    recommender.train()

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)   # (B, T)
            labels = batch["labels"].to(device)         # (B,)

            # Build augmented views for each sequence in the batch
            aug1_list, aug2_list = [], []
            for seq in input_ids:
                a1, a2 = augmenter(seq)
                aug1_list.append(a1)
                aug2_list.append(a2)

            aug1 = torch.stack(aug1_list).to(device)    # (B, T)
            aug2 = torch.stack(aug2_list).to(device)

            # SSL contrastive loss
            H1 = encoder(aug1)
            H2 = encoder(aug2)
            z1, z2 = H1[:, -1, :], H2[:, -1, :]
            loss_ssl = contrastive_loss(z1, z2)

            # Next-item recommendation loss
            logits = recommender(input_ids)
            loss_rec = F.cross_entropy(logits, labels)

            loss = loss_rec + loss_ssl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"[LACLRec] epoch {epoch + 1}/{epochs}  loss={total_loss / max(steps, 1):.4f}")