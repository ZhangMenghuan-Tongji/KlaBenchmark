from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_with_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class AutoKlaEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        num_layers: int = 12,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pad_id = int(pad_id)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        batch_size, length = x_ids.shape
        pos = torch.arange(length, device=x_ids.device).unsqueeze(0).expand(batch_size, length)
        hidden = self.token_emb(x_ids) + self.pos_emb(pos)
        src_key_padding_mask = x_ids.eq(self.pad_id)
        if src_key_padding_mask.shape[1] > 0:
            src_key_padding_mask[:, 0] = False
        return self.encoder(hidden, src_key_padding_mask=src_key_padding_mask)


class AutoKlaClassifier(nn.Module):
    def __init__(self, encoder: AutoKlaEncoder, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x_ids)
        cls_hidden = hidden[:, 0, :]
        return self.mlp(cls_hidden)


def build_model(config, vocab_size: int, pad_id: int) -> AutoKlaClassifier:
    encoder = AutoKlaEncoder(
        vocab_size=vocab_size,
        max_len=1 + int(config.seq_len),
        d_model=int(config.d_model),
        n_heads=int(config.n_heads),
        dim_ff=int(config.dim_ff),
        num_layers=int(config.num_layers),
        dropout=float(config.dropout),
        pad_id=int(pad_id),
    )
    return AutoKlaClassifier(encoder=encoder, d_model=int(config.d_model), dropout=float(config.dropout))
