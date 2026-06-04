from __future__ import annotations

import atexit
import os
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch

from configs import Config, METHOD_NAME, config_and_overrides_from_args, make_predict_parser, merge_config
from dataset import (
    build_feature_matrices_for_inference,
    prepare_sequences,
    read_tsv,
    seqs_to_lstm_ids,
)
from model import (
    batched_model_predict,
    build_feature_model,
    build_lstm_model,
    build_meta_model,
    esm2_predict,
    load_esm2_model,
)
from utils import (
    TeeRunLog,
    build_prediction_dataframe,
    compute_metrics,
    ensure_dir,
    load_json,
    pick_device,
    save_json,
    save_predictions,
    timestamp,
)


def load_training_defaults(checkpoint_path: str) -> dict:
    if not checkpoint_path:
        return {}

    checkpoint = Path(checkpoint_path).resolve()
    output_root = checkpoint.parent.parent if checkpoint.parent.name == "model" else checkpoint.parent
    overrides = {
        "checkpoint_path": str(checkpoint),
        "run_meta_path": str(output_root / "model" / "extra" / "run_meta.json"),
        "gps_encoder_path": str(output_root / "model" / "extra" / "gps_encoder.pt"),
        "lstm_model_path": str(output_root / "model" / "extra" / "lstm.pth"),
        "lstm_vocab_path": str(output_root / "model" / "extra" / "lstm_vocab.json"),
        "meta_feature_order_path": str(output_root / "model" / "extra" / "meta_feature_order.json"),
        "feature_model_dir": str(output_root / "model" / "extra" / "feature_dnns"),
        "esm2_finetuned_dir": str(output_root / "model" / "extra" / "esm2_finetuned"),
        "esm2_state_dict_path": str(output_root / "model" / "extra" / "esm2_state_dict.pth"),
    }

    run_config_path = output_root / "run_config.json"
    if run_config_path.exists():
        run_config = load_json(run_config_path)
        skip_keys = {"output_dir", "out_root", "run_name"}
        for key in Config.__dataclass_fields__:
            if key in run_config and key not in skip_keys:
                overrides[key] = run_config[key]

    run_meta_path = output_root / "model" / "extra" / "run_meta.json"
    if run_meta_path.exists():
        run_meta = load_json(run_meta_path)
        path_meta = run_meta.get("paths", {})
        config_meta = run_meta.get("config", {})
        for key in Config.__dataclass_fields__:
            if key in config_meta and key not in {"output_dir", "out_root", "run_name"}:
                overrides[key] = config_meta[key]
        for field in [
            "gps_encoder_path",
            "lstm_model_path",
            "lstm_vocab_path",
            "meta_feature_order_path",
            "feature_model_dir",
            "esm2_finetuned_dir",
            "esm2_state_dict_path",
            "run_meta_path",
            "checkpoint_path",
        ]:
            if field in path_meta:
                overrides[field] = path_meta[field]
    return overrides


def resolve_output_dir(config) -> Path:
    if config.output_dir:
        return Path(config.output_dir).resolve()
    checkpoint = Path(config.checkpoint_path).resolve()
    base_dir = checkpoint.parent.parent if checkpoint.parent.name == "model" else checkpoint.parent
    return (base_dir / f"predict_{timestamp()}").resolve()


def load_feature_models(feature_paths: dict[str, str], feature_model_dir: Path, device: torch.device):
    models = {}
    for feature_name, feature_path in feature_paths.items():
        x = np.load(feature_path, mmap_mode="r")
        model = build_feature_model(feature_name, in_size=x.shape[1]).to(device)
        state_dict = torch.load(feature_model_dir / f"{feature_name}.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models[feature_name] = model
    return models


def predict() -> None:
    parser = make_predict_parser()
    args = parser.parse_args()
    provisional_config, file_overrides, cli_overrides = config_and_overrides_from_args(args)
    artifact_overrides = load_training_defaults(provisional_config.checkpoint_path)
    config = merge_config(Config(), artifact_overrides, file_overrides, cli_overrides)

    if not config.checkpoint_path:
        raise ValueError("--checkpoint_path is required for prediction.")
    if not config.test_tsv:
        raise ValueError("--test_tsv/--input_tsv is required for prediction.")

    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print(f"[HybridKla] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    output_dir = resolve_output_dir(config)
    config.output_dir = str(output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "cache")

    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    device = pick_device(config.device)
    raw_input_df = pd.read_csv(config.test_tsv, sep="\t")
    input_df = read_tsv(config.test_tsv, require_label=False)
    if len(raw_input_df) != len(input_df):
        raise RuntimeError("Raw input TSV and validated input TSV have different row counts; prediction alignment is unsafe.")

    sequence_views = prepare_sequences(input_df, config.seq_len, split_label="predict")
    gps_encoder = None
    if config.enable_gps:
        if not config.gps_encoder_path:
            raise ValueError("GPS encoder path is required when GPS is enabled.")
        gps_encoder = torch.load(config.gps_encoder_path, map_location="cpu")

    feature_paths = build_feature_matrices_for_inference(
        seqs=sequence_views["feature"],
        cfg=config,
        cache_dir=output_dir / "cache" / "features",
        gps_encoder=gps_encoder,
    )
    feature_order = load_json(Path(config.meta_feature_order_path))
    feature_models = load_feature_models(feature_paths, Path(config.feature_model_dir), device=device)
    feature_probabilities = {}
    for feature_name in feature_order:
        model = feature_models[feature_name]
        feature_x = np.load(feature_paths[feature_name], mmap_mode="r")
        feature_probabilities[feature_name] = batched_model_predict(
            model,
            feature_x,
            device=device,
            batch_size=config.batch_size,
            dtype=torch.float32,
        )

    lstm_vocab = load_json(Path(config.lstm_vocab_path))
    lstm_model = build_lstm_model(vocab_size=max(lstm_vocab.values()) + 1, seq_len=config.seq_len).to(device)
    lstm_model.load_state_dict(torch.load(config.lstm_model_path, map_location=device))
    lstm_model.eval()
    lstm_ids = seqs_to_lstm_ids(sequence_views["model"], lstm_vocab)
    lstm_prob = batched_model_predict(
        lstm_model,
        lstm_ids,
        device=device,
        batch_size=config.batch_size,
        dtype=torch.long,
        postprocess=lambda logits: torch.softmax(logits, dim=1)[:, 1],
    )

    esm2_model, esm2_tokenizer = load_esm2_model(config.esm2_dir, config.esm2_state_dict_path)
    esm2_prob = esm2_predict(esm2_model, esm2_tokenizer, sequence_views["esm2"], device=device, batch_size=config.batch_size)

    meta_input = np.stack([esm2_prob, lstm_prob, *[feature_probabilities[name] for name in feature_order]], axis=1).astype(np.float32)
    meta_model = build_meta_model(in_size=meta_input.shape[1]).to(device)
    meta_model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    meta_model.eval()
    meta_prob = batched_model_predict(
        meta_model,
        meta_input,
        device=device,
        batch_size=config.batch_size,
        dtype=torch.float32,
    )

    if len(meta_prob) != len(input_df):
        raise RuntimeError(f"Prediction count mismatch: pred={len(meta_prob)} input={len(input_df)}")

    prediction_df = build_prediction_dataframe(
        raw_input_df.reset_index(drop=True),
        test_score=meta_prob,
        threshold=float(config.decision_threshold),
        esm2_prob=esm2_prob,
        lstm_prob=lstm_prob,
        feature_probabilities=feature_probabilities,
        feature_order=feature_order,
    )
    predictions_path = output_dir / "predictions.tsv"
    save_predictions(predictions_path, prediction_df)
    save_json(output_dir / "predict_config.json", config.to_dict())

    if "Label" in input_df.columns:
        metrics = compute_metrics(
            input_df["Label"].to_numpy(dtype=np.int64),
            meta_prob,
            threshold=float(config.decision_threshold),
        )
        save_json(
            output_dir / "predict_metrics.json",
            {
                "method": METHOD_NAME,
                "primary_result": "meta",
                "metrics": metrics,
                "paths": {
                    "checkpoint": config.checkpoint_path,
                    "predictions": str(predictions_path),
                },
            },
        )

    print(f"[HybridKla] Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    predict()
