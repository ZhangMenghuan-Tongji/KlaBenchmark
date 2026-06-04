from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from utils import align_raw_sequence, log_seq_len_adjustments, raw_sequence_length

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_ALLOWED = set(AA20)
AA_EXTRA = set("BZUOJ")


def load_tsv_rows(path: Path, require_label: bool = True) -> Tuple[List[Dict[str, str]], List[str], np.ndarray | None, List[str]]:
    rows: List[Dict[str, str]] = []
    seqs: List[str] = []
    labels: List[int] = []

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        if "Sequence" not in fieldnames:
            raise ValueError(f"TSV must contain column Sequence: {path}")
        if require_label and "Label" not in fieldnames:
            raise ValueError(f"TSV must contain columns Sequence and Label: {path}")

        for line_no, row in enumerate(reader, start=2):
            row = {key: ("" if value is None else str(value)) for key, value in row.items()}
            rows.append(row)
            seq = row.get("Sequence", "")
            seqs.append(str(seq))
            if "Label" in fieldnames:
                label = row.get("Label", "")
                try:
                    y = int(str(label).strip())
                except Exception as exc:
                    raise ValueError(f"Invalid Label at line {line_no} in {path}: {label!r}") from exc
                if y not in (0, 1):
                    raise ValueError(f"Labels must be binary 0/1. Got {y} at line {line_no} in {path}")
                labels.append(y)
    label_array = np.asarray(labels, dtype=np.int32) if labels else None
    return rows, seqs, label_array, fieldnames


def write_tsv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_sequence_chars(sequence: str) -> str:
    output = []
    for char in sequence:
        if char == "_" or char in AA20 or char == "X":
            output.append(char)
        else:
            output.append("X")
    return "".join(output)


def clean_sequence(raw: str, fixed_len: int = 31) -> str:
    aligned, _ = align_raw_sequence(
        raw,
        fixed_len,
        normalize_chars=_normalize_sequence_chars,
    )
    return aligned


def build_vocab() -> Dict[str, int]:
    return {
        "_": 0,
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "X": 21,
    }


def encode_sequences(
    seqs: Iterable[str],
    vocab: Dict[str, int],
    fixed_len: int,
    *,
    method_label: str = "DeepKla",
    split_label: str = "",
    warn: bool = True,
) -> np.ndarray:
    unk = vocab.get("X", 1)
    seqs_list = list(seqs)
    if warn:
        log_seq_len_adjustments(
            [raw_sequence_length(seq) for seq in seqs_list],
            int(fixed_len),
            method_label=method_label,
            split_label=split_label,
        )
    x = np.zeros((len(seqs_list), fixed_len), dtype=np.int32)
    for i, seq in enumerate(seqs_list):
        clean = clean_sequence(seq, fixed_len=fixed_len)
        x[i, :] = [vocab.get(char, unk) for char in clean]
    return x


def take_first(seqs: List[str], labels: np.ndarray, n: int) -> Tuple[List[str], np.ndarray]:
    if n and n > 0:
        return seqs[:n], labels[:n]
    return seqs, labels
