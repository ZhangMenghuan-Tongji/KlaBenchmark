# KlaBenchmark

KlaBenchmark is a benchmark-oriented code and dataset collection for lysine lactylation (Kla) site prediction. It provides standardized datasets and reproducible implementations of seven published Kla predictors:

- ABFF-Kla
- Auto-Kla
- DeepKla
- DeepKlapred
- HybridKla
- PBertKla
- PCBert-Kla

The repository supports downloading pretrained weights and benchmark data from [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main), running prediction with `predict.py`, and retraining with `train.py` under comparable evaluation protocols.

## Repository Layout

```text
KlaBenchmark/
├── checkpoint/       # Pretrained model checkpoints and preprocessing metadata
├── dataset/          # Complete benchmark datasets and dataset documentation
├── methods/          # Model code and usage guide
├── report/           # Reporting and plotting utilities
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -r methods/<method>/requirements.txt
```

See `methods/README.md` for details.

## Quick Start

Download model weights from [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main), then place each method's weights and metadata into the matching folder under `checkpoint/<method>/`.
After the checkpoint files are in place, run:

```bash
python methods/<method>/predict.py \
  --checkpoint_path <path/to/downloaded/checkpoint> \
  --input_tsv <path/to/input.tsv> \
  --output_dir <path/to/output>
```

## Retraining

```bash
python methods/<method>/train.py \
  --train_tsv <path/to/train.tsv> \
  --val_tsv <path/to/val.tsv> \
  --test_tsv <path/to/test.tsv> \
  --run_name <run_name>
```

## Documentation

- **`dataset/README.md`** — [Hugging Face data release](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main), required columns, benchmark partitions, label definitions
- **`methods/README.md`** — installation, prediction and training workflows, benchmark scenarios, external model resources, reporting guidelines
- **`report/plot_roc_prc.py`** — utility for plotting ROC and PRC curves from benchmark outputs

## License

Code in this repository is released under the MIT License. Datasets and third-party model weights may be subject to separate terms; see `dataset/README.md` and `methods/README.md`.

## Authors

- Yiqi Li
- Menghuan Zhang

## Citation

Please cite the following paper if you use KlaBenchmark:
