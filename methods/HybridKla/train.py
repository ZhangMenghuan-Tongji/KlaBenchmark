from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

from configs import METHOD_NAME, config_and_overrides_from_args, make_train_parser
from dataset import (
    NumpyDataset,
    build_feature_matrices,
    build_lstm_vocab,
    prepare_sequences,
    read_tsv,
    seqs_to_lstm_ids,
)
from model import (
    batched_model_predict,
    build_esm2_model,
    build_feature_model,
    build_lstm_model,
    build_meta_model,
    esm2_predict,
    stack_features,
)
from utils import (
    TeeRunLog,
    build_prediction_dataframe,
    build_run_paths,
    build_standard_metrics_payload,
    compute_metrics,
    default_run_name,
    ensure_dir,
    log,
    pick_device,
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


def train_binary_dnn(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    patience: int = 5,
) -> Tuple[nn.Module, Dict[str, float], Optional[int]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()
    train_loader = DataLoader(NumpyDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    best_auroc = -1.0
    best_state = None
    best_epoch = -1
    best_metrics: Optional[Dict[str, float]] = None
    last_val_pred = np.zeros((len(val_y),), dtype=np.float32)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            prediction = model(xb)
            loss = loss_fn(prediction, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu().item())
            num_batches += 1

        model.eval()
        val_pred = batched_model_predict(
            model,
            val_x,
            device=device,
            batch_size=batch_size,
            dtype=torch.float32,
        )
        last_val_pred = val_pred
        metrics_epoch = compute_metrics(val_y, val_pred)
        auroc = float(metrics_epoch["auc_roc"])
        avg_loss = running_loss / max(1, num_batches)
        log(
            f"Epoch {epoch + 1:03d}/{epochs} | loss={avg_loss:.6f} | "
            f"val_auroc={metrics_epoch['auc_roc']:.6f} val_acc={metrics_epoch['accuracy']:.6f} "
            f"val_mcc={metrics_epoch['mcc']:.6f}"
        )
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = metrics_epoch
            log(f"  -> best updated: best_val_auroc={best_auroc:.6f} (epoch={best_epoch + 1})")
        if epoch - best_epoch >= patience:
            log(f"  -> early stop: patience={patience}, stop at epoch={epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_metrics is None:
        best_metrics = compute_metrics(val_y, last_val_pred)
    best_metrics["best_val_auroc"] = best_auroc
    return model, best_metrics, (best_epoch + 1 if best_epoch >= 0 else None)


def train_lstm(
    train_seqs: List[str],
    train_y: np.ndarray,
    val_seqs: List[str],
    val_y: np.ndarray,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    out_path: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, int], Optional[int]]:
    trans = build_lstm_vocab()
    vocab_size = max(trans.values()) + 1
    model = build_lstm_model(vocab_size=vocab_size, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    x_train = seqs_to_lstm_ids(train_seqs, trans)
    x_val = seqs_to_lstm_ids(val_seqs, trans)

    class SeqDataset(Dataset):
        def __init__(self, x: np.ndarray, y: np.ndarray):
            self.x = torch.tensor(x, dtype=torch.long)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int):
            return self.x[idx], self.y[idx]

    train_loader = DataLoader(SeqDataset(x_train, train_y), batch_size=batch_size, shuffle=True)

    best_auroc = -1.0
    best_state = None
    best_epoch = -1
    best_metrics: Optional[Dict[str, float]] = None
    last_prob = np.zeros((len(val_y),), dtype=np.float32)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu().item())
            num_batches += 1

        model.eval()
        prob = batched_model_predict(
            model,
            x_val,
            device=device,
            batch_size=batch_size,
            dtype=torch.long,
            postprocess=lambda logits: torch.softmax(logits, dim=1)[:, 1],
        )
        last_prob = prob
        metrics_epoch = compute_metrics(val_y, prob)
        auroc = float(metrics_epoch["auc_roc"])
        avg_loss = running_loss / max(1, num_batches)
        log(
            f"[LSTM] Epoch {epoch + 1:03d}/{epochs} | loss={avg_loss:.6f} | "
            f"val_auroc={metrics_epoch['auc_roc']:.6f} val_acc={metrics_epoch['accuracy']:.6f} "
            f"val_mcc={metrics_epoch['mcc']:.6f}"
        )
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = metrics_epoch
            log(f"  -> [LSTM] best updated: best_val_auroc={best_auroc:.6f} (epoch={best_epoch + 1})")
        if epoch - best_epoch >= 5:
            log(f"  -> [LSTM] early stop: patience=5, stop at epoch={epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if out_path is not None:
        torch.save(model.state_dict(), out_path)
    if best_metrics is None:
        best_metrics = compute_metrics(val_y, last_prob)
    best_metrics["best_val_auroc"] = best_auroc
    return model, best_metrics, trans, (best_epoch + 1 if best_epoch >= 0 else None)


def train_esm2(
    train_seqs: List[str],
    train_y: np.ndarray,
    val_seqs: List[str],
    val_y: np.ndarray,
    cfg,
    out_dir: Path,
    *,
    save_model: bool = True,
    work_dir: Optional[Path] = None,
) -> Tuple[nn.Module, object, Dict[str, float], Optional[int]]:
    work_dir = Path(work_dir) if work_dir is not None else Path(out_dir)
    model, tokenizer = build_esm2_model(cfg.esm2_dir)

    class HFDataset(torch.utils.data.Dataset):
        def __init__(self, seqs: List[str], labels: np.ndarray):
            self.seqs = seqs
            self.labels = labels.astype(int).tolist()

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int):
            return {"sequence": self.seqs[idx], "labels": self.labels[idx]}

    def collate_fn(batch):
        seqs = [item["sequence"] for item in batch]
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        encoded = tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")
        encoded["labels"] = labels
        return encoded

    def compute_hf_metrics(eval_pred):
        logits, labels = eval_pred
        probabilities = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        return {
            "auprc": float(np.nan_to_num(compute_metrics(labels, probabilities)["auc_pr"])),
            "auroc": float(np.nan_to_num(compute_metrics(labels, probabilities)["auc_roc"])),
        }

    class HumanReadableCallback(TrainerCallback):
        def __init__(self):
            self.best_auroc = -1.0
            self.best_epoch = None

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return
            eval_auroc = metrics.get("eval_auroc")
            eval_loss = metrics.get("eval_loss")
            epoch = state.epoch
            if eval_auroc is not None:
                msg = f"[ESM2] Epoch {epoch:.2f} | eval_auroc={float(eval_auroc):.6f}"
                if eval_loss is not None:
                    msg += f" eval_loss={float(eval_loss):.6f}"
                log(msg)
                if float(eval_auroc) > self.best_auroc:
                    self.best_auroc = float(eval_auroc)
                    self.best_epoch = epoch
                    log(f"  -> [ESM2] best updated: best_eval_auroc={self.best_auroc:.6f}")

        def on_train_end(self, args, state, control, **kwargs):
            log(f"[ESM2] training finished: best_eval_auroc={self.best_auroc:.6f}")

    per_device_bs = int(cfg.batch_size)
    grad_accum = 1
    training_args = TrainingArguments(
        output_dir=str(work_dir / "hf_ckpt"),
        evaluation_strategy="epoch",
        save_strategy=("epoch" if bool(save_model) else "no"),
        logging_strategy="epoch",
        remove_unused_columns=False,
        load_best_model_at_end=bool(save_model),
        metric_for_best_model="auroc",
        greater_is_better=True,
        learning_rate=cfg.esm2_lr,
        warmup_steps=int(cfg.esm2_warmup_steps),
        warmup_ratio=0.0,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=cfg.epochs,
        optim="adamw_torch",
        weight_decay=cfg.esm2_weight_decay,
        report_to=[],
        fp16=False,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=HFDataset(train_seqs, train_y),
        eval_dataset=HFDataset(val_seqs, val_y),
        data_collator=collate_fn,
        compute_metrics=compute_hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    human_callback = HumanReadableCallback()
    trainer.add_callback(human_callback)
    trainer.train()

    if bool(save_model):
        ensure_dir(out_dir)
        trainer.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

    prediction = trainer.predict(HFDataset(val_seqs, val_y))
    probabilities = torch.softmax(torch.tensor(prediction.predictions), dim=1)[:, 1].numpy()
    metrics = compute_metrics(val_y, probabilities)
    metrics["best_val_auroc"] = float(metrics["auc_roc"])
    best_epoch = int(human_callback.best_epoch) if human_callback.best_epoch is not None else None
    return trainer.model, tokenizer, metrics, best_epoch


def train_feature_models(
    feats: Dict[str, Dict[str, str]],
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg,
    out_dir: Path,
    device: torch.device,
) -> Tuple[Dict[str, nn.Module], Dict[str, Dict[str, float]], Dict[str, Optional[int]]]:
    models: Dict[str, nn.Module] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    best_epochs: Dict[str, Optional[int]] = {}
    ensure_dir(out_dir)
    for feature_name, splits in feats.items():
        x_train = np.load(splits["train"], mmap_mode="r")
        x_val = np.load(splits["val"], mmap_mode="r")
        model = build_feature_model(feature_name, in_size=x_train.shape[1])
        model, feature_metrics, best_epoch = train_binary_dnn(
            model=model,
            train_x=x_train,
            train_y=y_train,
            val_x=x_val,
            val_y=y_val,
            device=device,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.feature_lr,
            patience=5,
        )
        torch.save(model.state_dict(), out_dir / f"{feature_name}.pth")
        models[feature_name] = model
        metrics[feature_name] = feature_metrics
        best_epochs[feature_name] = best_epoch
    return models, metrics, best_epochs


def predict_feature_models(
    models: Dict[str, nn.Module],
    feats: Dict[str, Dict[str, str]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    output: Dict[str, Dict[str, np.ndarray]] = {}
    for feature_name, model in models.items():
        model.eval().to(device)
        output[feature_name] = {}
        for split, path in feats[feature_name].items():
            x = np.load(path, mmap_mode="r")
            output[feature_name][split] = batched_model_predict(
                model,
                x,
                device=device,
                batch_size=batch_size,
                dtype=torch.float32,
            )
    return output


def train() -> None:
    parser = make_train_parser()
    args = parser.parse_args()
    config, _, _ = config_and_overrides_from_args(args)

    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print(f"[HybridKla] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    set_seed(config.seed)
    output_dir = resolve_output_dir(config)
    config.output_dir = str(output_dir)
    dataset_name = Path(config.train_tsv).resolve().parent.name
    device = pick_device(config.device)

    paths = build_run_paths(output_dir)
    for directory in [
        paths.output_dir,
        paths.model_dir,
        paths.model_extra_dir,
        paths.results_dir,
        paths.cache_dir,
        paths.feature_cache_dir,
        paths.feature_model_dir,
    ]:
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

    train_df = read_tsv(config.train_tsv)
    val_df = read_tsv(config.val_tsv)
    test_df = read_tsv(config.test_tsv)

    log("========== Dataset Summary ==========")
    log(f"train: n={len(train_df):,} | label={train_df['Label'].value_counts().to_dict()}")
    log(f"val  : n={len(val_df):,} | label={val_df['Label'].value_counts().to_dict()}")
    log(f"test : n={len(test_df):,} | label={test_df['Label'].value_counts().to_dict()}")
    log(f"Expected sequence length: {config.seq_len}")

    save_json(
        paths.dataset_stats_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "train": split_stats("train", train_df["Label"].to_numpy(dtype=np.int64)),
            "val": split_stats("val", val_df["Label"].to_numpy(dtype=np.int64)),
            "test": split_stats("test", test_df["Label"].to_numpy(dtype=np.int64)),
        },
    )

    train_sequences = prepare_sequences(train_df, config.seq_len, split_label="train")
    val_sequences = prepare_sequences(val_df, config.seq_len, split_label="val")
    test_sequences = prepare_sequences(test_df, config.seq_len, split_label="test")

    y_train = train_df["Label"].to_numpy(dtype=np.int64)
    y_val = val_df["Label"].to_numpy(dtype=np.int64)
    y_test = test_df["Label"].to_numpy(dtype=np.int64)

    log("========== Handcrafted Feature Encoding ==========")
    feats, preprocess = build_feature_matrices(
        train_seqs=train_sequences["feature"],
        val_seqs=val_sequences["feature"],
        test_seqs=test_sequences["feature"],
        cfg=config,
        cache_dir=paths.feature_cache_dir,
    )
    if "gps" in preprocess:
        torch.save(preprocess["gps"]["encoder"], paths.gps_encoder_path)

    set_seed(config.seed)
    log("========== Train 6 Feature DNNs ==========")
    feature_models, feature_val_metrics, feature_best_epochs = train_feature_models(
        feats=feats,
        y_train=y_train,
        y_val=y_val,
        cfg=config,
        out_dir=paths.feature_model_dir,
        device=device,
    )
    feat_probs = predict_feature_models(feature_models, feats, device=device, batch_size=config.batch_size)

    log("========== Train LSTM ==========")
    lstm_model, lstm_val_metrics, lstm_vocab, lstm_best_epoch = train_lstm(
        train_seqs=train_sequences["model"],
        train_y=y_train,
        val_seqs=val_sequences["model"],
        val_y=y_val,
        device=device,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        epochs=config.epochs,
        out_path=paths.lstm_model_path,
        lr=config.lstm_lr,
    )
    save_json(paths.lstm_vocab_path, lstm_vocab)

    def lstm_predict(split_sequences: List[str]) -> np.ndarray:
        lstm_model.eval().to(device)
        x = seqs_to_lstm_ids(split_sequences, lstm_vocab)
        return batched_model_predict(
            lstm_model,
            x,
            device=device,
            batch_size=config.batch_size,
            dtype=torch.long,
            postprocess=lambda logits: torch.softmax(logits, dim=1)[:, 1],
        )

    lstm_probs = {
        "train": lstm_predict(train_sequences["model"]),
        "val": lstm_predict(val_sequences["model"]),
        "test": lstm_predict(test_sequences["model"]),
    }

    log("========== Train ESM2 ==========")
    esm2_model, esm2_tokenizer, esm2_val_metrics, esm2_best_epoch = train_esm2(
        train_seqs=train_sequences["esm2"],
        train_y=y_train,
        val_seqs=val_sequences["esm2"],
        val_y=y_val,
        cfg=config,
        out_dir=paths.esm2_finetuned_dir,
        save_model=True,
        work_dir=paths.model_extra_dir,
    )
    torch.save(esm2_model.state_dict(), paths.esm2_state_dict_path)
    esm2_probs = {
        "train": esm2_predict(esm2_model, esm2_tokenizer, train_sequences["esm2"], device=device, batch_size=config.batch_size),
        "val": esm2_predict(esm2_model, esm2_tokenizer, val_sequences["esm2"], device=device, batch_size=config.batch_size),
        "test": esm2_predict(esm2_model, esm2_tokenizer, test_sequences["esm2"], device=device, batch_size=config.batch_size),
    }

    log("========== Train Meta-model ==========")
    feature_order = sorted(list(feat_probs.keys()))
    X_meta = stack_features(esm2_probs, lstm_probs, feat_probs, feature_order=feature_order)
    meta_model = build_meta_model(in_size=X_meta["train"].shape[1])
    meta_model, meta_val_metrics, meta_best_epoch = train_binary_dnn(
        model=meta_model,
        train_x=X_meta["train"],
        train_y=y_train,
        val_x=X_meta["val"],
        val_y=y_val,
        device=device,
        batch_size=config.batch_size,
        epochs=config.epochs,
        lr=config.meta_lr,
        patience=5,
    )
    torch.save(meta_model.state_dict(), paths.checkpoint_path)
    save_json(paths.meta_feature_order_path, feature_order)

    decision_threshold = float(config.decision_threshold)
    meta_model.eval().to(device)
    val_score_meta = batched_model_predict(
        meta_model,
        X_meta["val"],
        device=device,
        batch_size=config.batch_size,
        dtype=torch.float32,
    )
    val_metrics_meta = compute_metrics(y_val, val_score_meta, threshold=decision_threshold)
    meta_val_auroc = float(meta_val_metrics.get("best_val_auroc", val_metrics_meta["auc_roc"]))
    meta_val_mcc = float(val_metrics_meta["mcc"])

    log("========== Independent Test Evaluation (Meta-model) ==========")
    test_score = batched_model_predict(
        meta_model,
        X_meta["test"],
        device=device,
        batch_size=config.batch_size,
        dtype=torch.float32,
    )
    test_metrics = compute_metrics(y_test, test_score, threshold=decision_threshold)

    training_rows = [
        {"component": "Meta", **{k: float(v) for k, v in val_metrics_meta.items() if isinstance(v, (int, float, np.floating, np.integer))}},
        {"component": "ESM2", **{k: float(v) for k, v in esm2_val_metrics.items() if isinstance(v, (int, float, np.floating, np.integer))}},
        {"component": "LSTM", **{k: float(v) for k, v in lstm_val_metrics.items() if isinstance(v, (int, float, np.floating, np.integer))}},
    ]
    for feature_name, metrics in feature_val_metrics.items():
        training_rows.append(
            {"component": feature_name, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating, np.integer))}}
        )
    pd.DataFrame(training_rows).to_csv(paths.training_log_path, index=False)

    result_df = test_df.copy().reset_index(drop=True)
    if len(result_df) != len(test_sequences["raw"]):
        raise RuntimeError(f"Test row count changed: raw={len(test_sequences['raw'])} df={len(result_df)}")
    if result_df["Sequence"].astype(str).tolist() != test_sequences["raw"]:
        raise RuntimeError("Test Sequence order changed; refuse to write outputs to avoid misalignment.")

    for name, arr in [("prob_meta", test_score), ("prob_esm2", esm2_probs["test"]), ("prob_lstm", lstm_probs["test"])]:
        if len(arr) != len(result_df):
            raise RuntimeError(f"Prediction length mismatch: {name} pred={len(arr)} test={len(result_df)}")
    for feature_name in feature_order:
        if len(feat_probs[feature_name]["test"]) != len(result_df):
            raise RuntimeError(
                f"Prediction length mismatch: prob_{feature_name.lower()} pred={len(feat_probs[feature_name]['test'])} test={len(result_df)}"
            )

    prediction_df = build_prediction_dataframe(
        result_df,
        test_score=test_score,
        threshold=decision_threshold,
        esm2_prob=esm2_probs["test"],
        lstm_prob=lstm_probs["test"],
        feature_probabilities={feature_name: feat_probs[feature_name]["test"] for feature_name in feature_order},
        feature_order=feature_order,
    )
    save_predictions(paths.test_predictions_path, prediction_df)

    save_json(
        paths.preprocess_path,
        {
            "seq_len": config.seq_len,
            "meta_feature_order": feature_order,
            "features_enabled": sorted(list(feat_probs.keys())),
        },
    )
    save_json(
        paths.val_metrics_path,
        build_standard_metrics_payload(
            method=METHOD_NAME,
            raw_metrics=val_metrics_meta,
            primary_result="meta",
            best_epoch=meta_best_epoch,
            predictions_relpath="results/test_predictions.tsv",
            best_model_relpath="model/best_model.pth",
        ),
    )
    save_json(
        paths.test_metrics_path,
        build_standard_metrics_payload(
            method=METHOD_NAME,
            raw_metrics=test_metrics,
            primary_result="meta",
            best_epoch=meta_best_epoch,
            predictions_relpath="results/test_predictions.tsv",
            best_model_relpath="model/best_model.pth",
        ),
    )
    save_json(
        paths.run_meta_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "paths": {key: str(value) for key, value in vars(paths).items()},
            "config": config.to_dict(),
            "best_epochs": {
                "meta": meta_best_epoch,
                "esm2": esm2_best_epoch,
                "lstm": lstm_best_epoch,
                "features": feature_best_epochs,
            },
            "meta_val_auroc": meta_val_auroc,
            "meta_val_mcc": meta_val_mcc,
        },
    )

    print("\n================= DONE =================")
    print("Output directory:", str(output_dir))
    print("Independent test Meta-model metrics:")
    for key, value in test_metrics.items():
        print(f"- {key}: {value:.6f}")


if __name__ == "__main__":
    train()
