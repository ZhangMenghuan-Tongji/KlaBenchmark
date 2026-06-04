from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


FEATURE_HIDDEN_CONFIG = {
    "ACF": {"hidden": [1024, 512], "dropout": 0.3, "bn": True},
    "AAINDEX": {"hidden": [512, 256], "dropout": 0.4, "bn": True},
    "OBC": {"hidden": [2048, 512], "dropout": 0.5, "bn": True},
    "GPS": {"hidden": [512, 128], "dropout": 0.3, "bn": False},
    "CKSAAP": {"hidden": [768, 256], "dropout": 0.4, "bn": True},
    "PSEAAC": {"hidden": [64, 32], "dropout": 0.2, "bn": False},
}


class SimpleDNN(nn.Module):
    def __init__(self, in_size: int, hidden: List[int], dropout: float, use_bn: bool):
        super().__init__()
        layers: List[nn.Module] = []
        current = in_size
        for hidden_size in hidden:
            layers.append(nn.Linear(current, hidden_size))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current = hidden_size
        layers.append(nn.Linear(current, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, input_size: int, hidden_size: int, num_layers: int, max_seq_length: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.position_embedding = nn.Embedding(max_seq_length, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        emb = self.embedding(x) + self.position_embedding(pos_ids)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(emb, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out)


class CustomClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 2):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, 512)
        self.ln = nn.LayerNorm(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(512, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dense1(features)
        x = self.ln(x)
        x = torch.mean(x, dim=1)
        x = self.relu(x)
        x = self.dropout(x)
        return self.dense2(x)


class MetaModel(nn.Module):
    def __init__(self, in_size: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_size, 128)
        nn.init.ones_(self.fc1.bias)
        self.fc2 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(self.fc2(x))
        x = self.drop1(x)
        x = self.relu2(self.fc3(x))
        x = self.drop2(x)
        return self.sigmoid(self.fc4(x))


def build_feature_model(feature_name: str, in_size: int) -> SimpleDNN:
    config = FEATURE_HIDDEN_CONFIG.get(feature_name, {"hidden": [256, 64], "dropout": 0.3, "bn": True})
    return SimpleDNN(in_size=in_size, hidden=config["hidden"], dropout=config["dropout"], use_bn=config["bn"])


def build_lstm_model(vocab_size: int, seq_len: int) -> LSTMModel:
    return LSTMModel(vocab_size=vocab_size, input_size=128, hidden_size=256, num_layers=6, max_seq_length=seq_len)


def build_meta_model(in_size: int) -> MetaModel:
    return MetaModel(in_size=in_size)


def build_esm2_model(esm2_dir: str) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(esm2_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        esm2_dir,
        num_labels=2,
        hidden_dropout_prob=0.3,
        classifier_dropout=0.4,
    )
    model.classifier = CustomClassificationHead(model.config.hidden_size, num_labels=2)
    return model, tokenizer


def load_esm2_model(esm2_dir: str, state_dict_path: str | Path) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model, tokenizer = build_esm2_model(esm2_dir)
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model, tokenizer


def batched_model_predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    dtype: torch.dtype,
    postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> np.ndarray:
    batch_size = max(int(batch_size), 1)
    num_samples = int(len(x))
    if num_samples == 0:
        return np.zeros((0,), dtype=np.float32)

    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            xb = torch.as_tensor(np.asarray(x[start : start + batch_size]), dtype=dtype, device=device)
            pred = model(xb)
            if postprocess is not None:
                pred = postprocess(pred)
            outputs.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0,), dtype=np.float32)


def esm2_predict(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    seqs: List[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    model.to(device)
    outputs: List[np.ndarray] = []
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        outputs.append(probabilities)
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0,), dtype=np.float32)


def stack_features(
    esm2_probabilities: Dict[str, np.ndarray],
    lstm_probabilities: Dict[str, np.ndarray],
    feature_probabilities: Dict[str, Dict[str, np.ndarray]],
    feature_order: List[str],
) -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        cols = [esm2_probabilities[split], lstm_probabilities[split]]
        for feature_name in feature_order:
            cols.append(feature_probabilities[feature_name][split])
        output[split] = np.stack(cols, axis=1).astype(np.float32)
    return output


def save_json_text(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
