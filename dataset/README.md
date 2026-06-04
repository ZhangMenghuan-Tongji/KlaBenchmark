# KlaBenchmark Datasets

Benchmark TSV files for Kla site prediction are released at [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main). This directory may contain local copies for development; the canonical splits and sample counts match the manuscript.

## Download Full Dataset

The complete benchmark dataset is available from [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main). Download the full dataset archive, then unpack or copy each method-specific folder into this local `dataset/` directory.

Expected local layout:

```text
dataset/
├── ABFF-Kla/
├── Auto-Kla/
├── DeepKla/
├── DeepKlapred/
├── HybridKla/
├── PBertKla/
└── PCBert-Kla/
```

Each method folder should keep the partition structure from the [Hugging Face release](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main), for example:

```text
dataset/<method>/
├── protein_split_dataset/
│   ├── train.tsv
│   ├── val.tsv
│   └── test.tsv
├── original_model_dataset/
│   ├── train.tsv
│   ├── val.tsv
│   └── test.tsv
└── ...
```

After downloading, pass the corresponding TSV paths to each method:

```bash
python methods/<method>/train.py \
  --train_tsv dataset/<method>/<partition>/train.tsv \
  --val_tsv dataset/<method>/<partition>/val.tsv \
  --test_tsv dataset/<method>/<partition>/test.tsv
```

Keep the [Hugging Face release](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main) directory structure intact when possible. Some benchmark scenarios, such as fixed-window partitions, species-specific evaluation, and original-model datasets, depend on the exact partition folder names.

## Required Columns

For most predictors, each `train.tsv`, `val.tsv`, and `test.tsv` must include at least:

- **`Sequence`**: local sequence window centered on the candidate lysine
- **`Label`**: `1` for an experimentally identified Kla site, `0` for a background lysine site

**ABFF-Kla** is an exception: inputs also need structure-related fields such as **`Contact`**. See `methods/ABFF-Kla/README.md`.

## Optional Columns

Depending on the split, files may also include:

- `Site`: site-level identifier
- `ACC_ID`: protein accession (e.g. UniProt)
- `Mod_positions`: lysine position in the protein
- `Species`: source species
- `UniProtTaxID`: NCBI taxonomy identifier

## Benchmark Partitions

The [Hugging Face release](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main) includes the following directories. Each corresponds to a benchmark setting reported in the paper:

| Directory | Benchmark setting |
| --- | --- |
| `protein_split_dataset/` | Protein-level train/validation/test split |
| `sequence_split_dataset/` | Sequence-similarity-controlled split (50% identity) |
| `15aa_window_protein_split_dataset/` | Fixed 15 aa window, protein-level split |
| `31aa_window_protein_split_dataset/` | Fixed 31 aa window, protein-level split |
| `51aa_window_protein_split_dataset/` | Fixed 51 aa window, protein-level split |
| `species_dataset/` | Species-specific evaluation (`<Species>/train.tsv`, etc.) |
| `original_model_dataset/` | Datasets aligned with original model publications |

Each partition typically provides:

```text
train.tsv
val.tsv
test.tsv
```

### Selecting paths by scenario

When reproducing a specific figure or table in the manuscript, point `--train_tsv`, `--val_tsv`, and `--test_tsv` (in `methods/<method>/train.py` or `predict.py`) to the matching files below:

| Scenario | Typical data paths |
| --- | --- |
| Protein-level split | `protein_split_dataset/train.tsv`, etc. |
| Sequence-level split | `sequence_split_dataset/train.tsv`, etc. |
| Fixed window (15 / 31 / 51 aa) | `15aa_window_protein_split_dataset/...`, `31aa_window_protein_split_dataset/...`, or `51aa_window_protein_split_dataset/...` |
| Cross-species | `species_dataset/<Species>/train.tsv`, etc.; do not include non-target species in training or validation when evaluating generalization |
| Original-model setting | `original_model_dataset/train.tsv`, etc. |

For fixed-window partitions, use the window length that matches the dataset name when configuring the model (e.g. `--seq_len` where supported). Report window length in published results; model rankings can change with context size.

## Label Caveat

Background lysine sites (`Label = 0`) are **not** experimentally verified negatives. They are lysine residues from the same proteins that were not identified as Kla sites in the reanalyzed MS/MS data. Some may be true Kla sites under conditions not covered by the experiment. Treat benchmark tasks as **positive versus background** classification, not positive versus confirmed-negative.

## Usage Guidelines

- Keep official train/validation/test splits unchanged when reproducing published results.
- State which partition (protein-level, sequence-level, fixed window, species, or original-model) was used.
- Document any extra filtering, species subset, or custom window length.



## Redistribution

Benchmark data are for research use. When redistributing derivatives, comply with the terms of the original proteomics databases, UniProt, and source publications, and cite the KlaBenchmark manuscript and underlying studies.
