from __future__ import annotations

import atexit
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from configs import METHOD_NAME, Config, config_and_overrides_from_args, make_predict_parser, merge_config
from dataset import build_vocab, encode_with_cls, load_tsv
from model import build_model
from utils import (
    TeeRunLog,
    build_prediction_df,
    build_standard_metrics_payload,
    compute_metrics,
    default_run_name,
    ensure_dir,
    load_json,
    now_ts,
    pick_device,
    save_json,
    save_predictions,
)


def torch_load_state_dict(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=device)


def predict_scores(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, float | None]:
    model.eval()
    probs = []
    labels = []
    indices = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                xb, yb, ib = batch
                yb = yb.to(device)
            else:
                xb, ib = batch
                yb = None
            xb = xb.to(device)
            logits = model(xb)
            if criterion is not None and yb is not None:
                losses.append(float(criterion(logits, yb).item()))
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(float))
            indices.append(ib.detach().cpu().numpy().astype(np.int64))
            if yb is not None:
                labels.append(yb.detach().cpu().numpy().astype(int))
    y_prob = np.concatenate(probs) if probs else np.zeros((0,), dtype=float)
    y_idx = np.concatenate(indices) if indices else np.arange(y_prob.shape[0], dtype=np.int64)
    y_true = np.concatenate(labels) if labels else None
    loss = float(np.mean(losses)) if losses else None
    return y_prob, y_true, y_idx, loss


def load_training_defaults(config: Config) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else None
    candidate_files = []
    if config.run_meta_path:
        candidate_files.append(Path(config.run_meta_path))
    if checkpoint_path is not None:
        candidate_files.append(checkpoint_path.parent / "extra" / "run_meta.json")
        candidate_files.append(checkpoint_path.parent / "extra" / "best_meta.json")
        candidate_files.append(checkpoint_path.parent.parent / "run_config.json")
    for path in candidate_files:
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        if path.name == "run_config.json":
            for key, value in payload.items():
                if key in {"method", "dataset_name", "output_dir", "out_root", "run_name"}:
                    continue
                defaults[key] = value
            continue
        meta_config = payload.get("config")
        if isinstance(meta_config, dict):
            for key, value in meta_config.items():
                if key in {"output_dir", "out_root", "run_name"}:
                    continue
                defaults[key] = value
        for meta_key in ["preprocess_path", "best_model_path", "top3_meta_path"]:
            meta_value = payload.get(meta_key)
            if meta_value:
                defaults[meta_key.replace("best_model_path", "checkpoint_path")] = meta_value
        if payload.get("checkpoint_path"):
            defaults["checkpoint_path"] = payload["checkpoint_path"]
        if payload.get("preprocess_path"):
            defaults["preprocess_path"] = payload["preprocess_path"]
    return defaults


def infer_output_dir(config: Config) -> Path:
    if config.output_dir:
        return Path(config.output_dir)
    base_root = Path(config.out_root) if config.out_root else Path(__file__).resolve().parent / "runs"
    run_name = f"predict_{default_run_name(config.test_tsv)}"
    return base_root / run_name


def main() -> int:
    parser = make_predict_parser()
    args = parser.parse_args()
    config, _, cli_overrides = config_and_overrides_from_args(args)
    training_defaults = load_training_defaults(config)
    config = merge_config(config, training_defaults, cli_overrides)

    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required for prediction.")

    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print(f"[Auto-Kla] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    output_dir = infer_output_dir(config)
    results_dir = output_dir / "results"
    ensure_dir(output_dir)
    ensure_dir(results_dir)

    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    print(f"[{now_ts()}] Prediction output dir: {output_dir}", flush=True)

    preprocess_path = Path(config.preprocess_path) if config.preprocess_path else None
    if preprocess_path is None or not preprocess_path.exists():
        checkpoint_parent = Path(config.checkpoint_path).parent
        candidate = checkpoint_parent.parent / "preprocess.json"
        if candidate.exists():
            preprocess_path = candidate
    if preprocess_path is None or not preprocess_path.exists():
        raise FileNotFoundError("preprocess_path is required and could not be inferred from checkpoint_path.")

    preprocess = load_json(preprocess_path)
    vocab = preprocess.get("vocab") or build_vocab()
    config.seq_len = int(preprocess["seq_len"])

    input_df, input_seqs, input_labels = load_tsv(Path(config.test_tsv), require_label=False)
    x_input = encode_with_cls(
        input_seqs, vocab=vocab, seq_len=int(config.seq_len), split_label="predict"
    )

    if input_labels is not None:
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_input),
            torch.from_numpy(input_labels.astype(np.int64)),
            torch.arange(x_input.shape[0], dtype=torch.long),
        )
    else:
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_input),
            torch.arange(x_input.shape[0], dtype=torch.long),
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=True,
    )

    device = pick_device(config.device)
    print(f"[{now_ts()}] Device: {device}", flush=True)

    model = build_model(config, vocab_size=max(vocab.values()) + 1, pad_id=int(preprocess["pad_id"])).to(device)
    state_dict = torch_load_state_dict(Path(config.checkpoint_path), device=device)
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss() if input_labels is not None else None
    y_prob, y_true, y_idx, loss = predict_scores(model, loader, device, criterion)

    n = int(input_df.shape[0])
    if int(y_prob.shape[0]) != n:
        raise RuntimeError(f"Prediction count mismatch: pred={y_prob.shape[0]} input_rows={n}")
    if int(y_idx.shape[0]) != n:
        raise RuntimeError(f"Index count mismatch: idx={y_idx.shape[0]} input_rows={n}")
    if len(np.unique(y_idx)) != n:
        raise RuntimeError("Index not unique; prediction alignment cannot be guaranteed.")

    prob_aligned = np.empty((n,), dtype=float)
    prob_aligned[y_idx] = y_prob.astype(float)
    pred_df = build_prediction_df(input_df.reset_index(drop=True), prob_aligned, float(config.decision_threshold))
    save_predictions(results_dir / "test_predictions.tsv", pred_df)

    payload: dict[str, Any] = {
        "method": METHOD_NAME,
        "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
        "preprocess_path": str(preprocess_path.resolve()),
        "input_tsv": str(Path(config.test_tsv).resolve()),
        "output_dir": str(output_dir.resolve()),
        "prediction_path": str((results_dir / "test_predictions.tsv").resolve()),
        "config": asdict(config),
    }
    if loss is not None:
        payload["loss"] = float(loss)
    if y_true is not None:
        y_aligned = np.empty((n,), dtype=int)
        y_aligned[y_idx] = y_true.astype(int)
        metrics = compute_metrics(y_aligned, prob_aligned, threshold=float(config.decision_threshold))
        if loss is not None:
            metrics["loss"] = float(loss)
        metrics_payload = build_standard_metrics_payload(
            method=METHOD_NAME,
            n=n,
            raw_metrics=metrics,
            primary_result="checkpoint",
            monitor="provided_checkpoint",
            best_epoch=None,
            predictions_relpath="results/test_predictions.tsv",
            best_model_relpath=Path(config.checkpoint_path).name,
        )
        save_json(results_dir / "test_metrics.json", metrics_payload)
        payload["metrics_path"] = str((results_dir / "test_metrics.json").resolve())

    save_json(output_dir / "predict_config.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
