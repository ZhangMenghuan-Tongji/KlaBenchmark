from __future__ import annotations

import atexit
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from configs import METHOD_NAME, config_and_overrides_from_args, make_train_parser
from dataset import build_vocab, encode_with_cls, load_tsv
from model import build_model, cosine_with_warmup_lr
from utils import (
    TeeRunLog,
    build_prediction_df,
    build_run_paths,
    build_standard_metrics_payload,
    compute_metrics,
    default_run_name,
    ensure_dir,
    now_ts,
    pick_device,
    save_json,
    save_predictions,
    set_seed,
    split_stats,
    summarize_split,
)


def run_eval(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: List[float] = []
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    idxs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                xb, yb, ib = batch
            else:
                xb, yb = batch
                ib = None
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()))
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            ys.append(yb.detach().cpu().numpy().astype(int))
            ps.append(prob.astype(float))
            if ib is not None:
                idxs.append(ib.detach().cpu().numpy().astype(np.int64))
    y_true = np.concatenate(ys) if ys else np.zeros((0,), dtype=int)
    y_prob = np.concatenate(ps) if ps else np.zeros((0,), dtype=float)
    y_idx = np.concatenate(idxs) if idxs else np.arange(y_true.shape[0], dtype=np.int64)
    return (float(np.mean(losses)) if losses else 0.0), y_true, y_prob, y_idx


def run_train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    step0: int,
    grad_clip: float,
) -> tuple[float, int]:
    model.train()
    losses: List[float] = []
    step = step0
    for batch in loader:
        if len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
        step += 1
        lr = cosine_with_warmup_lr(step, total_steps=total_steps, warmup_steps=warmup_steps, base_lr=base_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return (float(np.mean(losses)) if losses else 0.0), step


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("state_dicts is empty")
    output: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        tensors = [state_dict[key].float() for state_dict in state_dicts]
        output[key] = torch.mean(torch.stack(tensors, dim=0), dim=0)
    return output


def torch_load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> int:
    parser = make_train_parser()
    args = parser.parse_args()
    config, file_overrides, cli_overrides = config_and_overrides_from_args(args)

    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print(f"[Auto-Kla] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    set_seed(int(config.seed))

    repo_root = Path(__file__).resolve().parent
    dataset_name = Path(config.train_tsv).parent.name or "run"
    run_name = config.run_name or default_run_name(config.train_tsv)
    output_dir = Path(config.output_dir) if config.output_dir else Path(config.out_root) / run_name
    paths = build_run_paths(output_dir)
    for path in [
        paths.output_dir,
        paths.model_dir,
        paths.checkpoint_dir,
        paths.model_extra_dir,
        paths.results_dir,
        paths.cache_dir,
    ]:
        ensure_dir(path)

    run_log = TeeRunLog(paths.output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    print(f"[{now_ts()}] Output dir: {paths.output_dir}", flush=True)
    save_json(paths.run_config_path, {**asdict(config), "method": METHOD_NAME, "dataset_name": dataset_name})

    train_df, train_seqs, y_train = load_tsv(Path(config.train_tsv), require_label=True)
    val_df, val_seqs, y_val = load_tsv(Path(config.val_tsv), require_label=True)
    test_df, test_seqs, y_test = load_tsv(Path(config.test_tsv), require_label=True)
    assert y_train is not None and y_val is not None and y_test is not None

    print(f"[{now_ts()}] {summarize_split('train', y_train)}", flush=True)
    print(f"[{now_ts()}] {summarize_split('val', y_val)}", flush=True)
    print(f"[{now_ts()}] {summarize_split('test', y_test)}", flush=True)

    save_json(
        paths.dataset_stats_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "train": split_stats("train", y_train),
            "val": split_stats("val", y_val),
            "test": split_stats("test", y_test),
        },
    )

    vocab = build_vocab()
    preprocess = {
        "seq_len": int(config.seq_len),
        "with_cls_len": 1 + int(config.seq_len),
        "vocab": vocab,
        "cls_token": "[CLS]",
        "cls_id": vocab["[CLS]"],
        "pad_token": "_",
        "pad_id": vocab["_"],
        "unknown_token": "X",
        "unknown_id": vocab["X"],
        "label_mapping": {"0": 0, "1": 1},
    }
    save_json(paths.preprocess_path, preprocess)

    x_train = encode_with_cls(
        train_seqs, vocab=vocab, seq_len=int(config.seq_len), split_label="train"
    )
    x_val = encode_with_cls(val_seqs, vocab=vocab, seq_len=int(config.seq_len), split_label="val")
    x_test = encode_with_cls(test_seqs, vocab=vocab, seq_len=int(config.seq_len), split_label="test")

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype(np.int64)),
        torch.arange(x_train.shape[0], dtype=torch.long),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(y_val.astype(np.int64)),
        torch.arange(x_val.shape[0], dtype=torch.long),
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(y_test.astype(np.int64)),
        torch.arange(x_test.shape[0], dtype=torch.long),
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=True,
    )

    device = pick_device(config.device)
    print(f"[{now_ts()}] Device: {device}", flush=True)

    model = build_model(config, vocab_size=max(vocab.values()) + 1, pad_id=vocab["_"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(config.epochs)
    warmup_steps = int(total_steps * float(config.warmup_ratio))
    print(
        f"[{now_ts()}] Train steps: steps/epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}",
        flush=True,
    )

    best = {"epoch": -1, "val_auroc": -1e9}
    ckpt_paths: List[Path] = []
    ckpt_meta: List[Dict[str, object]] = []
    epoch_rows: List[Dict[str, float]] = []
    step = 0
    bad_epochs = 0

    print(f"[{now_ts()}] Start training for {config.epochs} epochs ...", flush=True)
    for epoch in range(1, int(config.epochs) + 1):
        t0 = time.time()
        train_loss, step = run_train_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            base_lr=float(config.lr),
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            step0=step,
            grad_clip=float(config.grad_clip),
        )
        val_loss, yv_true, yv_prob, _ = run_eval(model, val_loader, device, criterion)
        val_stats = compute_metrics(yv_true, yv_prob, threshold=float(config.decision_threshold))
        duration = time.time() - t0

        print(
            f"[{now_ts()}] Epoch {epoch:03d}/{config.epochs} time={duration:.1f}s | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auroc={val_stats['auroc']:.4f} val_auprc={val_stats['auprc']:.4f} "
            f"val_acc={val_stats['acc']:.4f} val_mcc={val_stats['mcc']:.4f} "
            f"val_sn={val_stats['sn']:.4f} val_sp={val_stats['sp']:.4f} val_pre={val_stats['pre']:.4f}",
            flush=True,
        )

        ckpt_path = paths.checkpoint_dir / f"epoch{epoch:03d}_state_dict.pt"
        torch.save(model.state_dict(), ckpt_path)
        ckpt_paths.append(ckpt_path)
        ckpt_meta.append({"epoch": epoch, "val_loss": float(val_loss), "val_metrics": val_stats, "path": str(ckpt_path)})
        epoch_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auroc": float(val_stats["auroc"]),
                "val_auprc": float(val_stats["auprc"]),
                "val_accuracy": float(val_stats["acc"]),
                "val_precision": float(val_stats["pre"]),
                "val_recall": float(val_stats["sn"]),
                "val_specificity": float(val_stats["sp"]),
                "val_mcc": float(val_stats["mcc"]),
            }
        )

        if not np.isnan(val_stats["auroc"]) and float(val_stats["auroc"]) > float(best["val_auroc"]):
            best = {"epoch": epoch, "val_auroc": float(val_stats["auroc"])}
            save_json(paths.best_meta_path, best)
            print(f"[{now_ts()}] Best model updated: epoch={epoch} val_auroc={val_stats['auroc']:.4f}", flush=True)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if config.patience and bad_epochs >= int(config.patience):
                print(
                    f"[{now_ts()}] Early stopping triggered (patience={config.patience}) monitored by val_auroc. Stop at epoch={epoch}.",
                    flush=True,
                )
                break

    ckpt_meta_sorted = sorted(
        ckpt_meta,
        key=lambda item: float(item["val_metrics"]["auroc"]) if item.get("val_metrics") else -1e9,  # type: ignore[index]
        reverse=True,
    )
    top3 = ckpt_meta_sorted[:3]
    save_json(paths.top3_meta_path, top3)
    keep_ckpt_paths = {str(Path(item["path"]).resolve()) for item in top3}
    for ckpt_path in ckpt_paths:
        if str(ckpt_path.resolve()) not in keep_ckpt_paths and ckpt_path.exists():
            ckpt_path.unlink()

    pd.DataFrame(epoch_rows).to_csv(paths.training_log_path, index=False)
    print(
        f"[{now_ts()}] Training done. Best epoch={best['epoch']} best_val_auroc={best['val_auroc']:.4f} | soup_top3={[t['epoch'] for t in top3]}",
        flush=True,
    )

    def eval_state_dict(state_dict: Dict[str, torch.Tensor]) -> dict[str, object]:
        model.load_state_dict(state_dict)
        threshold = float(config.decision_threshold)
        val_loss, yv_true, yv_prob, _ = run_eval(model, val_loader, device, criterion)
        val_stats = compute_metrics(yv_true, yv_prob, threshold=threshold)
        val_stats["loss"] = float(val_loss)
        test_loss, yt_true, yt_prob, yt_idx = run_eval(model, test_loader, device, criterion)
        test_stats = compute_metrics(yt_true, yt_prob, threshold=threshold)
        test_stats["loss"] = float(test_loss)

        n = int(test_df.shape[0])
        if int(yt_prob.shape[0]) != n:
            raise RuntimeError(f"Prediction count mismatch: pred={yt_prob.shape[0]} test_rows={n}")
        if int(yt_idx.shape[0]) != n:
            raise RuntimeError(f"Index count mismatch: idx={yt_idx.shape[0]} test_rows={n}")
        yt_idx = yt_idx.astype(np.int64)
        if yt_idx.min(initial=0) < 0 or yt_idx.max(initial=-1) >= n:
            raise RuntimeError(f"Index out of range: min={yt_idx.min(initial=0)} max={yt_idx.max(initial=-1)} n={n}")
        if len(np.unique(yt_idx)) != n:
            raise RuntimeError("Index not unique; prediction alignment cannot be guaranteed.")

        prob_aligned = np.empty((n,), dtype=float)
        y_aligned = np.empty((n,), dtype=int)
        prob_aligned[yt_idx] = yt_prob.astype(float)
        y_aligned[yt_idx] = yt_true.astype(int)

        y_file = test_df["Label"].astype(int).to_numpy(dtype=int)
        if not np.array_equal(y_aligned, y_file):
            raise RuntimeError("Label alignment check failed: predicted-order labels != original test TSV labels.")
        return {
            "val_stats": val_stats,
            "test_stats": test_stats,
            "prob_aligned": prob_aligned,
        }

    best_sd = None
    if best["epoch"] > 0:
        best_ckpt = paths.checkpoint_dir / f"epoch{best['epoch']:03d}_state_dict.pt"
        best_sd = torch_load_state_dict(best_ckpt, device=device)
        torch.save(best_sd, paths.best_epoch_state_dict_path)
    else:
        print(f"[{now_ts()}] WARNING: best epoch not found, fallback to current model state.", flush=True)

    if len(top3) >= 1:
        state_dicts = [torch_load_state_dict(Path(item["path"]), device=device) for item in top3]  # type: ignore[index]
        soup_sd = average_state_dicts(state_dicts)
        torch.save(soup_sd, paths.best_model_path)
        print(f"[{now_ts()}] Saved soup checkpoint: {paths.best_model_path}", flush=True)
        primary_result = "soup"
        selected_best_epoch = None
        payload_source = eval_state_dict(soup_sd)
    else:
        print(f"[{now_ts()}] WARNING: not enough checkpoints for soup, fallback to best checkpoint.", flush=True)
        if best_sd is None:
            best_sd = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(best_sd, paths.best_model_path)
        primary_result = "best"
        selected_best_epoch = int(best["epoch"]) if best["epoch"] > 0 else None
        payload_source = eval_state_dict(best_sd)

    val_payload = build_standard_metrics_payload(
        method=METHOD_NAME,
        n=len(y_val),
        raw_metrics=payload_source["val_stats"],  # type: ignore[arg-type]
        primary_result=primary_result,
        monitor="val_auroc",
        best_epoch=selected_best_epoch,
        predictions_relpath="results/test_predictions.tsv",
        best_model_relpath="model/best_model.pt",
    )
    test_payload = build_standard_metrics_payload(
        method=METHOD_NAME,
        n=len(y_test),
        raw_metrics=payload_source["test_stats"],  # type: ignore[arg-type]
        primary_result=primary_result,
        monitor="val_auroc",
        best_epoch=selected_best_epoch,
        predictions_relpath="results/test_predictions.tsv",
        best_model_relpath="model/best_model.pt",
    )
    save_json(paths.val_metrics_path, val_payload)
    save_json(paths.test_metrics_path, test_payload)

    pred_df = build_prediction_df(test_df.reset_index(drop=True), payload_source["prob_aligned"], float(config.decision_threshold))  # type: ignore[arg-type]
    save_predictions(paths.test_predictions_path, pred_df)

    save_json(
        paths.run_meta_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "run_name": run_name,
            "output_dir": str(paths.output_dir),
            "best_model_path": str(paths.best_model_path),
            "best_epoch_state_dict_path": str(paths.best_epoch_state_dict_path),
            "preprocess_path": str(paths.preprocess_path),
            "top3_meta_path": str(paths.top3_meta_path),
            "best_meta_path": str(paths.best_meta_path),
            "results_dir": str(paths.results_dir),
            "val_metrics_path": str(paths.val_metrics_path),
            "test_metrics_path": str(paths.test_metrics_path),
            "test_predictions_path": str(paths.test_predictions_path),
            "primary_result": primary_result,
            "selected_best_epoch": selected_best_epoch,
            "config": asdict(config),
            "config_file_overrides": file_overrides,
            "config_cli_overrides": cli_overrides,
            "splits": {
                "train_rows": int(train_df.shape[0]),
                "val_rows": int(val_df.shape[0]),
                "test_rows": int(test_df.shape[0]),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
