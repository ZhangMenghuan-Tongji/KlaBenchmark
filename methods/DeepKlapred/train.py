from __future__ import annotations

import atexit
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data

from configs import METHOD_NAME, config_and_overrides_from_args, make_train_parser
from dataset import (
    BenchmarkDataset,
    build_embedding_inputs,
    build_residue2idx,
    load_tsv,
    make_features_for_split,
)
from model import DeepKlapredLogits
from utils import (
    TeeRunLog,
    build_prediction_dataframe,
    build_run_paths,
    build_standard_metrics_payload,
    compute_metrics,
    default_run_name,
    ensure_dir,
    now_ts,
    pick_device,
    run_predict_proba_with_index,
    save_json,
    save_predictions,
    set_seed,
    split_stats,
)


def resolve_output_dir(config) -> Path:
    if config.output_dir:
        return Path(config.output_dir).resolve()
    run_name = config.run_name or default_run_name(config.train_tsv)
    config.run_name = run_name
    return (Path(config.out_root) / run_name).resolve()


def run_eval(
    model: nn.Module,
    loader: Data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float = 0.5,
):
    model.eval()
    losses = []
    all_true = []
    all_prob = []
    with torch.no_grad():
        for _idx, input_ids, desc_feats, labels in loader:
            input_ids = input_ids.to(device)
            desc_feats = desc_feats.to(device)
            labels = labels.to(device)
            logits = model(input_ids, desc_feats)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_prob.append(prob)
            all_true.append(labels.detach().cpu().numpy())
    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    metrics = compute_metrics(y_true, y_prob, threshold=float(threshold))
    return float(np.mean(losses)) if losses else 0.0, metrics


def run_train_epoch(
    model: nn.Module,
    loader: Data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    losses = []
    for _idx, input_ids, desc_feats, labels in loader:
        input_ids = input_ids.to(device)
        desc_feats = desc_feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, desc_feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def train() -> None:
    parser = make_train_parser()
    args = parser.parse_args()
    config, _, _ = config_and_overrides_from_args(args)

    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print(f"[DeepKlapred] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    set_seed(config.seed)

    dataset_name = Path(config.train_tsv).resolve().parent.name or "data"
    output_dir = resolve_output_dir(config)
    config.output_dir = str(output_dir)
    paths = build_run_paths(output_dir)
    for directory in [paths.output_dir, paths.model_dir, paths.model_extra_dir, paths.results_dir, paths.cache_dir]:
        ensure_dir(directory)

    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    save_json(
        paths.run_config_path,
        {
            **config.to_dict(),
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
        },
    )

    device = pick_device(config.device)
    max_seq_len = int(config.seq_len)
    max_len_with_cls = max_seq_len + 1
    residue2idx = build_residue2idx()

    print(f"[{now_ts()}] Device: {device}")
    print(f"[{now_ts()}] seq_len={max_seq_len} (max_len_with_cls={max_len_with_cls})")
    print(f"[{now_ts()}] Loading TSVs...")
    train_df, train_seqs, train_y = load_tsv(config.train_tsv, require_label=True)
    val_df, val_seqs, val_y = load_tsv(config.val_tsv, require_label=True)
    test_df, test_seqs, test_y = load_tsv(config.test_tsv, require_label=True)
    assert train_y is not None and val_y is not None and test_y is not None

    print(
        f"[{now_ts()}] Sizes: train={len(train_seqs)} val={len(val_seqs)} test={len(test_seqs)} "
        f"(pos rate train={train_y.mean():.4f} val={val_y.mean():.4f} test={test_y.mean():.4f})"
    )
    save_json(
        paths.dataset_stats_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "train": split_stats("train", train_y),
            "val": split_stats("val", val_y),
            "test": split_stats("test", test_y),
        },
    )

    print(f"[{now_ts()}] Building embedding inputs...")
    x_train_ids = build_embedding_inputs(train_seqs, max_len_with_cls, split_label="train")
    x_val_ids = build_embedding_inputs(val_seqs, max_len_with_cls, split_label="val")
    x_test_ids = build_embedding_inputs(test_seqs, max_len_with_cls, split_label="test")

    print(f"[{now_ts()}] Extracting 739-d descriptor features (train)...")
    x_train_desc, feature_columns = make_features_for_split(
        train_seqs, feature_columns=None, raw_seq_len=max_seq_len
    )
    print(f"[{now_ts()}] Extracting 739-d descriptor features (val/test) with train columns...")
    x_val_desc, _ = make_features_for_split(
        val_seqs, feature_columns=feature_columns, raw_seq_len=max_seq_len
    )
    x_test_desc, _ = make_features_for_split(
        test_seqs, feature_columns=feature_columns, raw_seq_len=max_seq_len
    )

    print(
        f"[{now_ts()}] Feature shapes: "
        f"train_ids={x_train_ids.shape} train_desc={x_train_desc.shape} | "
        f"val_ids={x_val_ids.shape} val_desc={x_val_desc.shape} | "
        f"test_ids={x_test_ids.shape} test_desc={x_test_desc.shape}"
    )

    save_json(
        paths.preprocess_path,
        {
            "max_seq_len": max_seq_len,
            "max_len_with_cls": max_len_with_cls,
            "seq_len": max_seq_len,
            "residue2idx": residue2idx,
            "feature_columns": feature_columns,
            "train_tsv": config.train_tsv,
            "val_tsv": config.val_tsv,
            "test_tsv": config.test_tsv,
        },
    )

    train_ds = BenchmarkDataset(x_train_ids, x_train_desc, train_y)
    val_ds = BenchmarkDataset(x_val_ids, x_val_desc, val_y)
    test_ds = BenchmarkDataset(x_test_ids, x_test_desc, test_y)

    train_loader = Data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = Data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = Data.DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model = DeepKlapredLogits(
        vocab_size=max(residue2idx.values()) + 1,
        seq_feature_dim=x_train_desc.shape[1],
        max_len=max_len_with_cls,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr))

    best = {"epoch": -1, "auroc": -1e18}
    epoch_rows: List[Dict[str, float]] = []
    print(f"[{now_ts()}] Start training for {config.epochs} epochs...")
    bad_epochs = 0
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_metrics = run_eval(model, val_loader, device, criterion, threshold=0.5)
        elapsed = time.time() - t0

        print(
            f"[{now_ts()}] Epoch {epoch:03d}/{config.epochs} "
            f"time={elapsed:.1f}s | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_metrics.acc:.4f} val_f1={val_metrics.f1:.4f} "
            f"val_mcc={val_metrics.mcc:.4f} val_auroc={val_metrics.auroc:.4f} val_auprc={val_metrics.auprc:.4f} "
            f"val_sn={val_metrics.sn:.4f} val_sp={val_metrics.sp:.4f}"
        )
        epoch_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auroc": float(val_metrics.auroc),
                "val_auprc": float(val_metrics.auprc),
                "val_accuracy": float(val_metrics.acc),
                "val_f1": float(val_metrics.f1),
                "val_mcc": float(val_metrics.mcc),
                "val_recall": float(val_metrics.sn),
                "val_specificity": float(val_metrics.sp),
            }
        )

        val_auroc_cmp = val_metrics.auroc if (val_metrics.auroc == val_metrics.auroc) else -1e18
        if val_auroc_cmp > best["auroc"]:
            best = {"epoch": epoch, "auroc": float(val_auroc_cmp)}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics.__dict__,
                    "vocab_size": max(residue2idx.values()) + 1,
                    "seq_feature_dim": int(x_train_desc.shape[1]),
                    "max_len": int(max_len_with_cls),
                },
                paths.checkpoint_path,
            )
            bad_epochs = 0
            print(
                f"[{now_ts()}] Best model updated: epoch={epoch} val_auroc={val_metrics.auroc:.4f} saved={paths.checkpoint_path}"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(config.patience):
                print(
                    f"[{now_ts()}] Early stopping triggered: no val_auroc improvement for "
                    f"{config.patience} epochs (best_epoch={best['epoch']} best_val_auroc={best['auroc']:.6f})"
                )
                break

    print(f"[{now_ts()}] Training done. Best epoch={best['epoch']} best_val_auroc={best['auroc']:.6f}")
    pd.DataFrame(epoch_rows).to_csv(paths.training_log_path, index=False)

    ckpt = torch.load(paths.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    decision_cutoff = float(config.decision_threshold)
    _val_loss_best, val_metrics_best = run_eval(model, val_loader, device, criterion, threshold=decision_cutoff)
    test_loss, test_metrics = run_eval(model, test_loader, device, criterion, threshold=decision_cutoff)

    _idx, y_true_aligned, y_prob_pos_aligned = run_predict_proba_with_index(model, test_loader, device)
    assert len(test_df) == int(y_prob_pos_aligned.shape[0]), (
        f"Test TSV rows != predictions: tsv={len(test_df)} preds={int(y_prob_pos_aligned.shape[0])}"
    )
    if "Label" in test_df.columns:
        tsv_y = test_df["Label"].astype(int).to_numpy()
        assert np.array_equal(tsv_y, y_true_aligned), "Label mismatch between TSV order and dataloader order"

    test_out = build_prediction_dataframe(test_df, y_prob_pos_aligned, threshold=decision_cutoff)
    save_predictions(paths.test_predictions_path, test_out)

    _, val_true_aligned, val_prob_pos_aligned = run_predict_proba_with_index(model, val_loader, device)
    val_payload = build_standard_metrics_payload(
        method=METHOD_NAME,
        y_true=val_true_aligned,
        y_prob_pos=val_prob_pos_aligned,
        metrics_obj=val_metrics_best,
        threshold=decision_cutoff,
        primary_result="best",
        best_epoch=int(ckpt.get("epoch", -1)),
        predictions_relpath="results/test_predictions.tsv",
        best_model_relpath="model/best_model.pt",
    )
    test_payload = build_standard_metrics_payload(
        method=METHOD_NAME,
        y_true=y_true_aligned,
        y_prob_pos=y_prob_pos_aligned,
        metrics_obj=test_metrics,
        threshold=decision_cutoff,
        primary_result="best",
        best_epoch=int(ckpt.get("epoch", -1)),
        predictions_relpath="results/test_predictions.tsv",
        best_model_relpath="model/best_model.pt",
    )
    save_json(paths.val_metrics_path, val_payload)
    save_json(paths.test_metrics_path, test_payload)
    save_json(
        paths.run_meta_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "paths": {key: str(value) for key, value in vars(paths).items()},
            "config": config.to_dict(),
            "checkpoint": {
                "epoch": int(ckpt.get("epoch", -1)),
                "vocab_size": int(ckpt.get("vocab_size", max(residue2idx.values()) + 1)),
                "seq_feature_dim": int(ckpt.get("seq_feature_dim", x_train_desc.shape[1])),
                "max_len": int(ckpt.get("max_len", max_len_with_cls)),
            },
        },
    )

    print(
        f"[{now_ts()}] ===== Independent TEST =====\n"
        f"[{now_ts()}] test_loss={test_loss:.4f} "
        f"test_acc={test_metrics.acc:.4f} test_f1={test_metrics.f1:.4f} test_mcc={test_metrics.mcc:.4f} "
        f"test_auroc={test_metrics.auroc:.4f} test_auprc={test_metrics.auprc:.4f} "
        f"test_sn={test_metrics.sn:.4f} test_sp={test_metrics.sp:.4f}\n"
        f"[{now_ts()}] Predictions saved to: {paths.test_predictions_path}\n"
        f"[{now_ts()}] Artifacts saved to: {output_dir}"
    )


if __name__ == "__main__":
    train()
