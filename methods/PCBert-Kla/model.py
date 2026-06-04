from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

CLASSIFIER_INPUT_DIM = 1051


class Attention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.activation(self.proj(x))
        return weights * x


class PCBertKlaClassifier(nn.Module):
    """Classifier with the same feature-level attention and MLP stack as the source script."""

    def __init__(self, input_dim: int = CLASSIFIER_INPUT_DIM):
        super().__init__()
        self.dropout1 = nn.Dropout(p=0.1)
        self.att0 = Attention(input_dim)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(x)
        x = self.att0(x)
        x = self.dropout2(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x).squeeze(-1)


def truncate_bert_layers(bert, keep_bert_layers: int):
    if hasattr(bert, "encoder") and hasattr(bert.encoder, "layer"):
        del bert.encoder.layer[keep_bert_layers:]
    return bert


def build_tokenizer_and_backbone(config, cache_dir: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        config.protbert_dir,
        cache_dir=str(Path(cache_dir) / "hf_cache"),
        local_files_only=config.local_files_only,
    )
    try:
        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        special_tokens = 2
    max_length_tokens = int(config.seq_len) + special_tokens

    bert = AutoModel.from_pretrained(
        config.protbert_dir,
        cache_dir=str(Path(cache_dir) / "hf_cache"),
        local_files_only=config.local_files_only,
    )
    bert = truncate_bert_layers(bert, int(config.keep_bert_layers))
    bert = bert.to(device)
    return tokenizer, bert, max_length_tokens
