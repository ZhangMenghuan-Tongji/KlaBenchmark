from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import align_raw_sequence, log_seq_len_adjustments, raw_sequence_length

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def build_vocab() -> Dict[str, int]:
    vocab: Dict[str, int] = {"_": 0}
    for i, aa in enumerate(AA20, start=1):
        vocab[aa] = i
    vocab["X"] = 21
    vocab["[CLS]"] = 22
    return vocab


def _normalize_sequence_chars(sequence: str) -> str:
    output = []
    for ch in sequence:
        if ch == "_" or ch in AA20 or ch == "X":
            output.append(ch)
        else:
            output.append("X")
    return "".join(output)


def normalize_seq(raw: str, seq_len: int = 51) -> str:
    aligned, _ = align_raw_sequence(
        raw,
        seq_len,
        normalize_chars=_normalize_sequence_chars,
    )
    return aligned


def encode_with_cls(
    seqs: List[str],
    vocab: Dict[str, int],
    seq_len: int = 51,
    *,
    method_label: str = "Auto-Kla",
    split_label: str = "",
    warn: bool = True,
) -> np.ndarray:
    if warn:
        log_seq_len_adjustments(
            [raw_sequence_length(raw) for raw in seqs],
            int(seq_len),
            method_label=method_label,
            split_label=split_label,
        )
    cls_id = vocab["[CLS]"]
    unk_id = vocab.get("X", 1)
    x = np.zeros((len(seqs), 1 + seq_len), dtype=np.int64)
    for i, raw in enumerate(seqs):
        sequence = normalize_seq(raw, seq_len=seq_len)
        ids = [cls_id] + [vocab.get(ch, unk_id) for ch in list(sequence)]
        x[i, :] = np.asarray(ids, dtype=np.int64)
    return x


def load_tsv(path: Path, require_label: bool = True) -> tuple[pd.DataFrame, List[str], np.ndarray | None]:
    df = pd.read_csv(path, sep="\t")
    if "Sequence" not in df.columns:
        raise ValueError(f"TSV must contain column Sequence. Got: {list(df.columns)}")
    if require_label and "Label" not in df.columns:
        raise ValueError(f"TSV must contain columns Sequence and Label. Got: {list(df.columns)}")
    seqs = df["Sequence"].astype(str).tolist()
    y = df["Label"].astype(int).to_numpy(dtype=np.int64) if "Label" in df.columns else None
    return df, seqs, y
