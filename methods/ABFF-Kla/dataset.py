from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from utils import center_align_length, log_seq_len_adjustments, raw_sequence_length

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_ID = {"_": 0}
for index, amino_acid in enumerate(AA20, start=1):
    AA_TO_ID[amino_acid] = index
AA_TO_ID["X"] = 21
VOCAB_SIZE = 22

AA3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def normalize_seq(sequence: str) -> str:
    if sequence is None or (isinstance(sequence, float) and np.isnan(sequence)):
        return ""
    return str(sequence).strip().upper()


def pad_or_trim(sequence: str, length: int, pad_char: str = "_") -> str:
    return center_align_length(normalize_seq(sequence), length, pad_char=pad_char)


def encode_seq(sequence: str, length: int) -> np.ndarray:
    sequence = pad_or_trim(normalize_seq(sequence), length, "_")
    ids = np.zeros((length,), dtype=np.int32)
    for index, char in enumerate(sequence):
        ids[index] = AA_TO_ID.get(char, 21)
    return ids


def load_split_tsv(path: str, require_label: bool = True):
    df = pd.read_csv(path, sep="\t", dtype=str)
    if ("Context" not in df.columns) and ("Sequence" in df.columns):
        df["Context"] = df["Sequence"]
    required = ["Context", "Contact"]
    if require_label:
        required.append("Label")
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column {column} in {path}. Found: {list(df.columns)}")
    ctx = df["Context"].fillna("").astype(str).values
    cont = df["Contact"].fillna("").astype(str).values
    labels = df["Label"].astype(int).values if "Label" in df.columns else None
    return df, ctx, cont, labels


def encode_pair(
    ctx_arr,
    cont_arr,
    length: int = 31,
    *,
    method_label: str = "ABFF-Kla",
    split_label: str = "",
    warn: bool = True,
):
    if warn:
        log_seq_len_adjustments(
            [raw_sequence_length(sequence) for sequence in ctx_arr],
            int(length),
            method_label=method_label,
            split_label=f"{split_label} Context".strip(),
        )
        cont_lengths = [raw_sequence_length(sequence) for sequence in cont_arr if normalize_seq(sequence) != ""]
        if cont_lengths:
            log_seq_len_adjustments(
                cont_lengths,
                int(length),
                method_label=method_label,
                split_label=f"{split_label} Contact".strip(),
            )
    acid_x = np.stack([encode_seq(sequence, length) for sequence in ctx_arr], axis=0)
    cont_fixed = []
    empty_count = 0
    for sequence in cont_arr:
        sequence = normalize_seq(sequence)
        if sequence == "":
            empty_count += 1
            sequence = "_" * length
        cont_fixed.append(sequence)
    cont_x = np.stack([encode_seq(sequence, length) for sequence in cont_fixed], axis=0)
    return acid_x, cont_x, empty_count


class AFContactSegmenter:
    def __init__(self, pdb_dir: Path, chain_id: str = "A", cutoff_a: float = 10.0, seg_len: int = 31):
        self.pdb_dir = Path(pdb_dir)
        self.chain_id = str(chain_id)
        self.cutoff_a = float(cutoff_a)
        self.seg_len = int(seg_len)
        self.half = self.seg_len // 2
        self._pdb_cache = {}
        self._pdb_miss = set()

    def load_af_pdb(self, acc: str):
        if acc in self._pdb_cache:
            return self._pdb_cache[acc]
        if acc in self._pdb_miss:
            return None

        acc = str(acc).strip().upper()
        acc_base = acc.split("-", 1)[0]
        pdb_path = None

        exact_patterns = [
            f"AF-{acc}-F1-model.pdb",
            f"AF-{acc}-F1-model_v6.pdb",
            f"AF-{acc}-F1-model_v5.pdb",
            f"AF-{acc}-F1-model_v4.pdb",
            f"{acc}.pdb",
        ]
        for name in exact_patterns:
            candidate = self.pdb_dir / name
            if candidate.exists():
                pdb_path = candidate
                break

        if pdb_path is None:
            exact_patterns_base = [
                f"AF-{acc_base}-F1-model.pdb",
                f"AF-{acc_base}-F1-model_v6.pdb",
                f"AF-{acc_base}-F1-model_v5.pdb",
                f"AF-{acc_base}-F1-model_v4.pdb",
                f"{acc_base}.pdb",
            ]
            for name in exact_patterns_base:
                candidate = self.pdb_dir / name
                if candidate.exists():
                    pdb_path = candidate
                    break

        if pdb_path is None:
            hits = sorted(self.pdb_dir.glob(f"AF-{acc}-F*-model*.pdb"))
            if not hits:
                hits = sorted(self.pdb_dir.glob(f"AF-{acc_base}-F*-model*.pdb"))
            if hits:
                pdb_path = hits[0]

        if pdb_path is None:
            hits = sorted(self.pdb_dir.glob(f"*{acc}*.pdb"))
            if not hits:
                hits = sorted(self.pdb_dir.glob(f"*{acc_base}*.pdb"))
            if hits:
                pdb_path = hits[0]

        if pdb_path is None:
            self._pdb_miss.add(acc)
            return None

        try:
            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(acc, str(pdb_path))
            model = structure[0]
            chain = model[self.chain_id]
        except Exception:
            self._pdb_miss.add(acc)
            return None

        pos_list, aa_list, ca_list = [], [], []
        for residue in chain:
            hetflag, resseq, _icode = residue.id
            if hetflag != " " or "CA" not in residue:
                continue
            resname = residue.get_resname()
            if resname not in AA3_TO_1:
                continue
            pos_list.append(int(resseq))
            aa_list.append(AA3_TO_1[resname])
            ca_list.append(residue["CA"].get_coord())

        if not pos_list:
            self._pdb_miss.add(acc)
            return None

        pos_arr = np.array(pos_list, dtype=int)
        aa_arr = np.array(aa_list, dtype="<U1")
        ca_arr = np.array(ca_list, dtype=float)
        self._pdb_cache[acc] = (pos_arr, aa_arr, ca_arr)
        return self._pdb_cache[acc]

    @staticmethod
    def map_modpos_to_index(pos_arr, aa_arr, mod_pos_1based: int):
        hits = np.where(pos_arr == int(mod_pos_1based))[0]
        if hits.size > 0:
            return int(hits[0])

        idx = int(mod_pos_1based) - 1
        if 0 <= idx < len(aa_arr) and aa_arr[idx] == "K":
            return idx
        return None

    def build_contact_segment(self, aa_arr, ca_arr, target_idx: int):
        size = len(aa_arr)
        if target_idx is None or not (0 <= target_idx < size):
            return None

        distances = np.linalg.norm(ca_arr - ca_arr[target_idx], axis=1)
        contact = np.where(distances < self.cutoff_a)[0]
        if target_idx not in contact:
            contact = np.append(contact, target_idx)

        contact_by_dist = contact[np.argsort(distances[contact])]
        max_take = min(len(contact_by_dist), self.seg_len * 3)
        chosen_sorted = np.sort(contact_by_dist[:max_take])
        if target_idx not in chosen_sorted:
            chosen_sorted = np.sort(np.append(chosen_sorted, target_idx))

        pos_in = int(np.where(chosen_sorted == target_idx)[0][0])
        left_need, right_need = self.half, self.half
        left = max(0, pos_in - left_need)
        right = min(len(chosen_sorted) - 1, pos_in + right_need)

        window = chosen_sorted[left : right + 1]
        while len(window) < self.seg_len:
            if left > 0:
                left -= 1
                window = np.insert(window, 0, chosen_sorted[left])
                continue
            if right < len(chosen_sorted) - 1:
                right += 1
                window = np.append(window, chosen_sorted[right])
                continue
            break

        if target_idx not in window:
            return None

        wpos = int(np.where(window == target_idx)[0][0])
        left_part = window[max(0, wpos - self.half) : wpos]
        right_part = window[wpos + 1 : wpos + 1 + self.half]

        seg_chars = []
        if len(left_part) < self.half:
            seg_chars.extend(["_"] * (self.half - len(left_part)))
        seg_chars.extend(aa_arr[left_part].tolist()[-self.half :])
        seg_chars.append(aa_arr[target_idx])
        seg_chars.extend(aa_arr[right_part].tolist()[: self.half])
        if len(right_part) < self.half:
            seg_chars.extend(["_"] * (self.half - len(right_part)))

        seg = "".join(seg_chars)
        if len(seg) != self.seg_len:
            seg = center_align_length(seg, self.seg_len, pad_char="_")
        return seg

    def process_tsv(self, in_tsv: str, out_tsv: str | Path, report_every: int = 2000):
        df = pd.read_csv(in_tsv, sep="\t", dtype=str)
        for column in ["ACC_ID", "Mod_positions"]:
            if column not in df.columns:
                raise ValueError(f"{in_tsv} is missing required column {column}; found {list(df.columns)}")
        if ("Context" not in df.columns) and ("Sequence" not in df.columns):
            raise ValueError(f"{in_tsv} must contain Context or Sequence; found {list(df.columns)}")

        ctx_series = df["Context"] if "Context" in df.columns else df["Sequence"]
        contact_segments = []
        stats = Counter()

        try:
            from tqdm import tqdm

            progress = tqdm(total=len(df), desc="cut_contact", unit="row", dynamic_ncols=True)
        except Exception:
            progress = None

        pending_updates = 0
        progress_update_every = 10000

        def flush_progress(force: bool = False):
            nonlocal pending_updates
            if pending_updates == 0:
                return
            if not force and pending_updates < progress_update_every:
                return
            if progress is not None:
                progress.set_postfix(
                    {"ok": stats["ok"], "no_pdb": stats["no_pdb"], "map_fail": stats["map_fail"], "notK": stats["not_K"]},
                    refresh=force,
                )
                progress.update(pending_updates)
            pending_updates = 0

        for _, row in df.iterrows():
            stats["total"] += 1
            acc = str(row["ACC_ID"]).strip()
            mod_pos_raw = row["Mod_positions"]
            seg = ""

            pdb_data = self.load_af_pdb(acc)
            if pdb_data is None:
                stats["no_pdb"] += 1
            else:
                if pd.isna(mod_pos_raw):
                    stats["bad_modpos"] += 1
                else:
                    try:
                        mod_pos = int(float(mod_pos_raw))
                    except Exception:
                        mod_pos = None
                        stats["bad_modpos"] += 1

                    if mod_pos is not None:
                        pos_arr, aa_arr, ca_arr = pdb_data
                        target_idx = self.map_modpos_to_index(pos_arr, aa_arr, mod_pos)
                        if target_idx is None:
                            stats["map_fail"] += 1
                        elif aa_arr[target_idx] != "K":
                            stats["not_K"] += 1
                        else:
                            seg_candidate = self.build_contact_segment(aa_arr, ca_arr, target_idx)
                            if seg_candidate is None:
                                stats["seg_none"] += 1
                            else:
                                seg = seg_candidate
                                if len(seg_candidate) == self.seg_len and seg_candidate[self.half] == "K":
                                    stats["ok"] += 1
                                else:
                                    stats["center_not_K"] += 1

            contact_segments.append(seg)
            pending_updates += 1
            flush_progress()

            if stats["total"] % report_every == 0:
                flush_progress(force=True)
                print(
                    f"\nprocessed={stats['total']}/{len(df)} | ok={stats['ok']} | no_pdb={stats['no_pdb']} | "
                    f"map_fail={stats['map_fail']} | not_K={stats['not_K']} | bad_modpos={stats['bad_modpos']} | "
                    f"center_not_K={stats['center_not_K']}"
                )

        flush_progress(force=True)
        if progress is not None:
            progress.close()

        df["Contact"] = contact_segments
        df["Context"] = ctx_series.fillna("")
        out_path = Path(out_tsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)
        return stats, str(out_path)


def ensure_with_contact_tsv(
    in_tsv: str,
    output_dir: Path,
    pdb_dir: str,
    chain_id: str = "A",
    cutoff_a: float = 10.0,
    seg_len: int = 31,
    report_every: int = 50000,
) -> str:
    in_tsv = str(in_tsv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    head = pd.read_csv(in_tsv, sep="\t", dtype=str, nrows=1)
    cols = set(head.columns)
    if ("Contact" in cols) and (("Context" in cols) or ("Sequence" in cols)):
        return in_tsv

    required_raw_common = {"ACC_ID", "Mod_positions"}
    has_ctx = ("Context" in cols) or ("Sequence" in cols)
    if (not required_raw_common.issubset(cols)) or (not has_ctx):
        raise ValueError(
            f"Input TSV is neither a with-contact TSV nor a raw TSV that can generate contact sequences. "
            f"Expected columns {sorted(required_raw_common)} plus Context/Sequence; got {list(head.columns)}; path={in_tsv}"
        )

    out_path = output_dir / f"{Path(in_tsv).stem}.with_contact.tsv"
    segmenter = AFContactSegmenter(
        pdb_dir=Path(pdb_dir),
        chain_id=chain_id,
        cutoff_a=cutoff_a,
        seg_len=seg_len,
    )
    stats, saved = segmenter.process_tsv(in_tsv, out_path, report_every=report_every)
    print(f"\n[contact] generated: {saved}")
    print(f"[contact] stats: {dict(stats)}\n")
    return saved
