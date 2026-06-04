from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import iFeatureOmegaCLI
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

from utils import align_raw_sequence, log_seq_len_adjustments, raw_sequence_length


def build_residue2idx() -> Dict[str, int]:
    return {
        "[PAD]": 0,
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
        "[CLS]": 22,
        "[SEP]": 23,
    }


def _normalize_embedding_chars(sequence: str) -> str:
    output = []
    for ch in sequence:
        if ch == "_" or ch in "ACDEFGHIKLMNPQRSTVWY" or ch == "X":
            output.append(ch)
        else:
            output.append("X")
    return "".join(output)


def clean_sequence_for_embedding(seq: str, raw_seq_len: int) -> List[int]:
    residue2idx = build_residue2idx()
    aligned, _ = align_raw_sequence(
        seq,
        int(raw_seq_len),
        normalize_chars=_normalize_embedding_chars,
    )
    return [residue2idx["[CLS]"]] + [residue2idx.get(ch, residue2idx["X"]) for ch in aligned]


def _normalize_descriptor_chars(sequence: str) -> str:
    output = []
    for ch in sequence:
        if ch in "ACDEFGHIKLMNPQRSTVWY" or ch == "X":
            output.append(ch)
        elif ch == "_":
            output.append("X")
        else:
            output.append("X")
    return "".join(output)


def clean_sequence_for_descriptor(seq: str, raw_seq_len: int) -> str:
    aligned, _ = align_raw_sequence(
        seq,
        int(raw_seq_len),
        normalize_chars=_normalize_descriptor_chars,
    )
    return aligned


def load_tsv(path: str | Path, require_label: bool = True) -> Tuple[pd.DataFrame, List[str], np.ndarray | None]:
    df = pd.read_csv(path, sep="\t")
    if "Sequence" not in df.columns:
        raise ValueError(f"TSV must contain the Sequence column, got: {df.columns.tolist()}")
    if require_label and "Label" not in df.columns:
        raise ValueError(f"TSV must contain Sequence and Label columns, got: {df.columns.tolist()}")
    seqs = df["Sequence"].astype(str).tolist()
    labels = df["Label"].astype(int).to_numpy() if "Label" in df.columns else None
    return df, seqs, labels


def write_fasta_like(seqs: List[str], path: str | Path, raw_seq_len: int) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for i, seq in enumerate(seqs):
            handle.write(f">seq_{i}\n")
            handle.write(f"{clean_sequence_for_descriptor(seq, raw_seq_len)}\n")


def _dde_feature_from_fasta_like(path: str | Path) -> pd.DataFrame:
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    codons = {
        "A": 4, "C": 2, "D": 2, "E": 2, "F": 2, "G": 4, "H": 2, "I": 3, "K": 2, "L": 6,
        "M": 1, "N": 2, "P": 4, "Q": 2, "R": 6, "S": 6, "T": 4, "V": 4, "W": 1, "Y": 2,
    }
    dipeptides = [a + b for a in amino_acids for b in amino_acids]
    tm = np.array([(codons[p[0]] / 61) * (codons[p[1]] / 61) for p in dipeptides], dtype=np.float64)
    aa2i = {aa: i for i, aa in enumerate(amino_acids)}

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    seqs = []
    for i in range(0, len(lines), 2):
        seq = lines[i + 1].strip().replace("-", "")
        seqs.append(seq)

    feats = np.zeros((len(seqs), 400), dtype=np.float64)
    for row, seq in enumerate(seqs):
        if len(seq) < 2:
            continue
        counts = np.zeros(400, dtype=np.float64)
        for j in range(len(seq) - 1):
            a, b = seq[j], seq[j + 1]
            if a in aa2i and b in aa2i:
                counts[aa2i[a] * 20 + aa2i[b]] += 1
        if counts.sum() > 0:
            counts = counts / counts.sum()
        tv = tm * (1 - tm) / (len(seq) - 1)
        dde = np.where(tv != 0, (counts - tm) / np.sqrt(tv), 0.0)
        feats[row] = dde

    return pd.DataFrame(feats, columns=[f"DDE{i + 1}" for i in range(400)])


def generate_descriptor_features(fasta_like_path: str | Path) -> pd.DataFrame:
    descriptors = ["QSOrder", "CTDC", "CTDT", "CTDD", "DistancePair"]
    parts = []
    for descriptor in descriptors:
        protein = iFeatureOmegaCLI.iProtein(str(fasta_like_path))
        protein.get_descriptor(descriptor)
        parts.append(protein.encodings.reset_index(drop=True))
    dde = _dde_feature_from_fasta_like(fasta_like_path).reset_index(drop=True)
    parts.append(dde)
    output = pd.concat(parts, axis=1)
    return output.select_dtypes(include=[np.number]).reset_index(drop=True)


def make_features_for_split(
    seqs: List[str],
    feature_columns: Optional[List[str]] = None,
    *,
    raw_seq_len: int,
) -> Tuple[np.ndarray, List[str]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        fasta_path = Path(temp_dir) / "seqs.fa"
        write_fasta_like(seqs, fasta_path, raw_seq_len=int(raw_seq_len))
        frame = generate_descriptor_features(fasta_path)
    if feature_columns is None:
        feature_columns = frame.columns.tolist()
    frame = frame.reindex(columns=feature_columns, fill_value=0.0)
    features = frame.to_numpy(dtype=np.float32, copy=True)
    return features, feature_columns


def build_embedding_inputs(
    seqs: List[str],
    max_len_with_cls: int,
    *,
    method_label: str = "DeepKlapred",
    split_label: str = "",
    warn: bool = True,
) -> np.ndarray:
    raw_seq_len = int(max_len_with_cls) - 1
    if warn:
        log_seq_len_adjustments(
            [raw_sequence_length(seq) for seq in seqs],
            raw_seq_len,
            method_label=method_label,
            split_label=split_label,
        )
    return np.stack([clean_sequence_for_embedding(seq, raw_seq_len) for seq in seqs], axis=0)


class BenchmarkDataset(Data.Dataset):
    def __init__(self, input_ids: np.ndarray, desc_feats: np.ndarray, labels: np.ndarray):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.desc_feats = torch.tensor(desc_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.tensor(idx, dtype=torch.long),
            self.input_ids[idx],
            self.desc_feats[idx],
            self.labels[idx],
        )
