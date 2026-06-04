from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils import align_sequence_batch

# ProteinBERT tokenize_seq prepends <START> and appends <END> (see proteinbert/tokenization.py).
PROTEINBERT_ADDED_TOKENS_PER_SEQ = 2


def proteinbert_token_len(raw_seq_len: int) -> int:
    """Total token positions for ProteinBERT encode_X / create_model (raw length + START/END)."""
    return max(int(raw_seq_len), 0) + PROTEINBERT_ADDED_TOKENS_PER_SEQ


def read_raw_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t").reset_index(drop=True)


def load_tsv(path: Path, *, require_label: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "Sequence" not in df.columns:
        raise ValueError(f"TSV must contain column 'Sequence': {path}")
    if require_label and "Label" not in df.columns:
        raise ValueError(f"TSV must contain columns 'Sequence' and 'Label': {path}")

    keep_columns = ["Sequence"]
    if "Label" in df.columns:
        keep_columns.append("Label")
    df = df[keep_columns].copy()

    if require_label:
        label_num = pd.to_numeric(df["Label"], errors="coerce")
        if label_num.isna().any():
            bad_rows = (label_num.isna()).to_numpy().nonzero()[0].tolist()
            head_rows = bad_rows[:20]
            raise ValueError(
                f"Label contains missing/non-numeric values at rows (0-based): {head_rows}"
                + (" ..." if len(bad_rows) > 20 else "")
                + f" in {path}"
            )
        df["Label"] = label_num.astype(int)
        unique_labels = set(df["Label"].unique().tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be binary 0/1. Got: {sorted(list(unique_labels))} in {path}")
    elif "Label" in df.columns:
        label_num = pd.to_numeric(df["Label"], errors="coerce")
        if not label_num.isna().all():
            df["Label"] = label_num.astype("Int64")

    return df.reset_index(drop=True)


def _proteinbert_compatible_chars(sequence: str) -> str:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    chars = []
    for ch in sequence:
        if ch in allowed or ch == "X":
            chars.append(ch)
        elif ch == "_":
            chars.append("X")
        else:
            chars.append("X")
    return "".join(chars)


def proteinbert_compatible_seqs(
    seqs: Iterable,
    seq_len: int,
    *,
    method_label: str = "PBertKla",
    split_label: str = "",
    warn: bool = True,
) -> list[str]:
    aligned = align_sequence_batch(
        seqs,
        int(seq_len),
        method_label=method_label,
        split_label=split_label,
        warn=warn,
    )
    return [_proteinbert_compatible_chars(seq) for seq in aligned]


def encode_sequences(
    input_encoder,
    seqs: Iterable,
    seq_len: int,
    *,
    method_label: str = "PBertKla",
    split_label: str = "",
    warn: bool = True,
):
    compatible = proteinbert_compatible_seqs(
        seqs,
        seq_len,
        method_label=method_label,
        split_label=split_label,
        warn=warn,
    )
    token_len = proteinbert_token_len(seq_len)
    return input_encoder.encode_X(compatible, token_len), compatible


def load_benchmark_triplet(train_path: Path, val_path: Path, test_path: Path):
    train_df = load_tsv(train_path, require_label=True)
    val_df = load_tsv(val_path, require_label=True)
    raw_test_df = read_raw_tsv(test_path)
    test_df = load_tsv(test_path, require_label=True)
    if len(raw_test_df) != len(test_df):
        raise RuntimeError(
            "The raw test TSV and the validated test TSV have different row counts; "
            "refuse to continue to avoid misaligned predictions."
        )
    return train_df, val_df, raw_test_df, test_df


def print_dataset_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    def summarize(df: pd.DataFrame) -> str:
        pos = int(df["Label"].sum())
        n = int(len(df))
        neg = n - pos
        lengths = df["Sequence"].astype(str).str.len()
        return (
            f"n={n} pos={pos} neg={neg} "
            f"len(min/med/max)={int(lengths.min())}/{int(lengths.median())}/{int(lengths.max())}"
        )

    print("[PBertKla] Dataset summary:")
    print(f"  - train: {summarize(train_df)}")
    print(f"  - val  : {summarize(val_df)}")
    print(f"  - test : {summarize(test_df)}")


def build_dataset_stats(method_name: str, dataset_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    return {
        "method": method_name,
        "dataset_name": dataset_name,
        "train": {
            "n": int(len(train_df)),
            "pos": int(train_df["Label"].sum()),
            "neg": int((1 - train_df["Label"]).sum()),
        },
        "val": {
            "n": int(len(val_df)),
            "pos": int(val_df["Label"].sum()),
            "neg": int((1 - val_df["Label"]).sum()),
        },
        "test": {
            "n": int(len(test_df)),
            "pos": int(test_df["Label"].sum()),
            "neg": int((1 - test_df["Label"]).sum()),
        },
        "seq_len_unique_train": sorted(train_df["Sequence"].astype(str).str.len().unique().tolist())[:20],
    }


def validate_prediction_scores(y_score: np.ndarray, expected_length: int) -> None:
    if int(len(y_score)) != int(expected_length):
        raise RuntimeError(
            "Prediction count does not match the input table row count; "
            "this suggests an encoding or inference alignment problem."
        )
    if not np.isfinite(y_score).all():
        raise RuntimeError("Prediction scores contain NaN/Inf; refuse to write outputs.")
