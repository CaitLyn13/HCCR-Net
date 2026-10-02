from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .condition_processing_controls import (
    CONTROL_METHODS,
    MAPPING_SEED,
    OCMM_EPOCHS,
    prepare_control_fold,
    save_processing_state,
)
from .data import load_target_rul
from .metrics import metric_dict
from .paths import DEFAULT_CMAPSS_DIR
from .protocol_v1_config import (
    DOMAINS,
    FROZEN_FIXED_EPOCHS,
    PROTOCOL_V1_NAME,
    PROTOCOL_V1_VERSION,
    REPORT_SEEDS,
    ProtocolV1Config,
)
from .protocol_v1_models import build_conditioncomp_rul_net
from .protocol_v1_train import train_final_p3


DEFAULT_OUTPUT = Path("results/workbench/condition_processing_controls_v1")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}.")


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


@torch.no_grad()
def _predict(model, windows: np.ndarray, batch_size: int, device: str):
    torch_device = torch.device(device)
    model.eval().to(torch_device)
    outputs: list[np.ndarray] = []
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    started = time.perf_counter()
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start : start + batch_size]).to(
            torch_device, non_blocking=torch_device.type == "cuda"
        )
        outputs.append(model(batch)["prediction"].cpu().numpy())
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - started
    return np.concatenate(outputs), 1000.0 * elapsed / max(1, len(windows))


def _load_model(checkpoint: Path, config: ProtocolV1Config, method: str, sensors: int):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    observed = (
        payload.get("condition_processing_method"),
        payload.get("config", {}).get("target_domain"),
        payload.get("config", {}).get("seed"),
        payload.get("selected_epoch"),
        payload.get("input_channels"),
    )
    expected = (method, config.target_domain, config.seed, FROZEN_FIXED_EPOCHS, sensors)
    if observed != expected:
        raise ValueError(f"Control checkpoint mismatch: {observed} != {expected}")
    model = build_conditioncomp_rul_net(
        sensors=sensors,
        window_size=config.window_size,
        retained_bins=config.slow_bins,
        representation_mode="condition_residual",
    )
    model.load_state_dict(payload["model_state"])
    return model


def execute(
    config: ProtocolV1Config,
    method: str,
    prepared_fold=None,
    force_train: bool = False,
) -> dict:
    if method not in CONTROL_METHODS:
        raise ValueError(f"method must be one of {CONTROL_METHODS}")
    if config.epochs != FROZEN_FIXED_EPOCHS:
        raise ValueError(f"Controls require exactly {FROZEN_FIXED_EPOCHS} epochs.")
    fold = prepared_fold or prepare_control_fold(config, method)
    run_dir = config.output_dir / method / config.target_domain / f"seed_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_processing_state(run_dir / "preprocessing.npz", fold.state)
    checkpoint = run_dir / "checkpoints" / "condition_processing_control.pt"
    sensors = fold.fit.windows.shape[2]
    if checkpoint.exists() and not force_train:
        model = _load_model(checkpoint, config, method, sensors)
    else:
        result = train_final_p3(
            fold.fit,
            config,
            FROZEN_FIXED_EPOCHS,
            checkpoint,
            representation_mode="condition_residual",
        )
        model = result.model
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload.update(
            {
                "method": "condition-processing-control",
                "condition_processing_method": method,
                "input_channels": sensors,
                "mapping_seed": MAPPING_SEED,
                "ocmm_epochs": OCMM_EPOCHS if method == "rafd_ocmm_style" else 0,
            }
        )
        torch.save(payload, checkpoint)

    predictions, latency = _predict(
        model, fold.target.windows, config.tta_inference_batch_size, config.device
    )
    before_labels = pd.DataFrame(
        {
            "engine_id": fold.target.engine,
            "cycle": fold.target.cycle,
            "is_final": fold.target.is_final,
            "prediction": predictions,
        }
    )
    before_labels.to_csv(run_dir / "predictions_before_labels.csv", index=False)
    final = (
        before_labels.loc[before_labels["is_final"]]
        .sort_values("engine_id")
        .reset_index(drop=True)
    )
    target_rul = load_target_rul(
        config.raw_dir, config.target_domain, expected_engines=len(final)
    )
    expected_engines = np.arange(1, len(target_rul) + 1, dtype=np.int64)
    if not np.array_equal(final.engine_id.to_numpy(np.int64), expected_engines):
        raise ValueError("Target engine IDs do not align with official RUL rows.")
    final["true_rul"] = target_rul
    final["error"] = final["prediction"] - final["true_rul"]
    final["method"] = method
    final.to_csv(run_dir / "engine_predictions.csv", index=False)
    error = final.error.to_numpy(np.float64)
    metrics = {
        **metric_dict(
            final.true_rul.to_numpy(np.float64),
            final.prediction.to_numpy(np.float64),
        ),
        "signed_bias": float(error.mean()),
        "worst_10pct_engine_rmse": float(
            np.sqrt(np.mean(np.sort(error**2)[-max(1, int(np.ceil(0.1 * len(error)))) :]))
        ),
        "window_latency_ms": latency,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    _save_json(run_dir / "metrics.json", metrics)
    audit = {
        "protocol": PROTOCOL_V1_NAME,
        "protocol_version": PROTOCOL_V1_VERSION,
        "experiment": "condition_processing_controls_v1",
        "method": method,
        "target_domain": config.target_domain,
        "seed": config.seed,
        "fixed_epochs": FROZEN_FIXED_EPOCHS,
        "input_channels": sensors,
        "source_fit_engines": list(fold.fit_engine_ids),
        "source_calibration_engines": list(fold.calibration_engine_ids),
        "partitions_disjoint": not bool(
            set(fold.fit_engine_ids) & set(fold.calibration_engine_ids)
        ),
        "healthy_rows": fold.state.healthy_rows,
        "mapping_seed": MAPPING_SEED,
        "ocmm_epochs": OCMM_EPOCHS if method == "rafd_ocmm_style" else 0,
        "target_predictions_saved_before_labels": True,
        "target_time_updates": 0,
        "checkpoint": str(checkpoint.resolve()),
        "config": {
            **asdict(config),
            "raw_dir": str(config.raw_dir),
            "output_dir": str(config.output_dir),
        },
        "metrics": metrics,
    }
    _save_json(run_dir / "control_audit.json", audit)
    return audit


def _config(args, target: str, seed: int) -> ProtocolV1Config:
    return ProtocolV1Config(
        target_domain=target,
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=seed,
        device=args.device,
        epochs=FROZEN_FIXED_EPOCHS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Source-only condition-processing controls")
    parser.add_argument("--target", choices=DOMAINS)
    parser.add_argument("--method", choices=CONTROL_METHODS)
    parser.add_argument("--seed", type=int, choices=REPORT_SEEDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_CMAPSS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        for method in CONTROL_METHODS:
            for target in DOMAINS:
                base = _config(args, target, REPORT_SEEDS[0])
                fold = prepare_control_fold(base, method)
                for seed in REPORT_SEEDS:
                    audit = execute(
                        _config(args, target, seed),
                        method,
                        prepared_fold=fold,
                        force_train=args.force_train,
                    )
                    print(method, target, seed, json.dumps(audit["metrics"]))
                del fold
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return
    if args.target is None or args.method is None or args.seed is None:
        raise SystemExit("Specify --target, --method and --seed, or use --all.")
    audit = execute(
        _config(args, args.target, args.seed),
        args.method,
        force_train=args.force_train,
    )
    print(json.dumps(audit["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
