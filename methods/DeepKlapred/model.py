from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout=0.2, max_len=max_len)
        self.bi_gru = nn.GRU(
            d_model,
            d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.src_emb(input_ids)
        x = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
        out, _ = self.bi_gru(x)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class CrossAttention(nn.Module):
    def __init__(self, in_dim1: int, in_dim2: int, k_dim: int = 32, v_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.proj_q = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        batch1, _ = x1.size()
        batch2, _ = x2.size()
        assert batch1 == batch2
        q = self.proj_q(x1).view(batch1, self.num_heads, self.k_dim).permute(0, 2, 1)
        k = self.proj_k(x2).view(batch1, self.num_heads, self.k_dim).permute(0, 2, 1)
        v = self.proj_v(x2).view(batch1, self.num_heads, self.v_dim).permute(0, 2, 1)
        attn = torch.matmul(q.transpose(1, 2), k) / (self.k_dim**0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(1, 2))
        out = out.transpose(1, 2).contiguous().view(batch1, -1)
        return self.proj_o(out)


class DeepKlapredLogits(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_feature_dim: int,
        max_len: int,
        d_model: int = 256,
        d_ff: int = 512,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.emb = EmbeddingLayer(vocab_size, d_model, max_len=max_len)
        self.transformer = nn.Sequential(*[TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc_desc = nn.Sequential(
            nn.Linear(seq_feature_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.cross_att = CrossAttention(d_model, d_model, num_heads=n_heads)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.6),
            nn.ReLU(),
        )
        self.out = nn.Linear(64, 2)

    def forward(self, input_ids: torch.Tensor, desc_feats: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        x = self.transformer(x)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        descriptor = self.fc_desc(desc_feats)
        fused = self.cross_att(pooled, descriptor)
        hidden = self.fc(fused)
        return self.out(hidden)
