"""Run MDMT under the frozen four-fold source-only C-MAPSS protocol.

Artifact contract: CODE-MDMT-RUN-001. The target covariate file is not
opened until source training and checkpoint persistence are complete.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import (
    SourceData,
    SourceStatistics,
    add_train_rul,
    fit_source_statistics,
    load_target_rul,
    load_test_domain,
    load_train_domain,
    make_source_windows,
    make_target_stream,
    normalize,
    normalize_settings,
    sensor_columns,
)
from .mdmt_baseline import (
    MDMT_BATCH_SIZE,
    MDMT_KERNEL_MULTIPLIERS,
    MDMT_PERTURBATION,
    MDMT_SOURCE_WEIGHT,
    MDMT_STAGE_EPOCHS,
    MDMT_STAGE_LEARNING_RATES,
    MDMT_TEMPERATURE,
    PAPER_PARAMETER_COUNT,
    REPRODUCED_PARAMETER_COUNT,
    MDMTBiLSTM,
    mdmt_stage_schedule,
    train_mdmt,
    trainable_parameter_count,
)
from .metrics import metric_dict
from .paths import DEFAULT_CMAPSS_DIR, WORKBENCH_RESULTS_DIR
from .protocol_v1_config import DOMAINS, REPORT_SEEDS, ProtocolV1Config
from .protocol_v1_data import split_fit_calibration_engines


PROTOCOL_NAME = "mdmt-unified-source-only-v1"


@dataclass(frozen=True)
class PreparedMDMTSources:
    fit: SourceData
    calibration: SourceData
    statistics: SourceStatistics
    source_domains: tuple[str, ...]
    fit_engine_ids: tuple[int, ...]
    calibration_engine_ids: tuple[int, ...]


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


def _save_statistics(path: Path, statistics: SourceStatistics) -> None:
    if path.exists():
        return
    np.savez_compressed(
        path,
        preprocessing=np.asarray(statistics.preprocessing),
        mean=statistics.mean,
        std=statistics.std,
        feature_columns=np.asarray(statistics.feature_columns),
        setting_mean=statistics.setting_mean,
        setting_std=statistics.setting_std,
        condition_coefficients=statistics.condition_coefficients,
        regime_centers=statistics.regime_centers,
        regime_mean=statistics.regime_mean,
        regime_std=statistics.regime_std,
    )


def _registered_config(
    config: ProtocolV1Config,
    reported_output_dir: Path | None = None,
) -> dict:
    """Only fields that are active in the MDMT reproduction are persisted."""
    return {
        "target_domain": config.target_domain,
        "raw_dir": str(config.raw_dir),
        "output_dir": str(reported_output_dir or config.output_dir),
        "seed": config.seed,
        "device": config.device,
        "window_size": config.window_size,
        "stride": 1,
        "rul_cap": config.rul_cap,
        "calibration_fraction": config.calibration_fraction,
        "split_seed": config.split_seed,
        "target_inference_batch_size": config.tta_inference_batch_size,
    }


def _subset(frame: pd.DataFrame, engines: np.ndarray) -> pd.DataFrame:
    return frame.loc[frame["unit_id"].isin(engines)].copy()


def _combine(parts: list[tuple[np.ndarray, ...]]) -> SourceData:
    raw_rul = np.concatenate([part[2] for part in parts])
    return SourceData(
        windows=np.concatenate([part[0] for part in parts]),
        rul=np.concatenate([part[1] for part in parts]),
        stage=np.zeros(len(raw_rul), dtype=np.int64),
        domain=np.concatenate([part[3] for part in parts]),
        engine=np.concatenate([part[4] for part in parts]),
        settings=np.concatenate([part[5] for part in parts]),
        cycle=np.concatenate([part[6] for part in parts]),
    )


def prepare_mdmt_sources(config: ProtocolV1Config) -> PreparedMDMTSources:
    """Prepare only source partitions; this function never opens target files."""
    config.validate()
    source_domains = tuple(domain for domain in DOMAINS if domain != config.target_domain)
    fit_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    fit_engine_ids: list[int] = []
    calibration_engine_ids: list[int] = []
    for domain_id, domain in enumerate(source_domains):
        frame = add_train_rul(load_train_domain(config.raw_dir, domain), config.rul_cap)
        fit_units, calibration_units = split_fit_calibration_engines(
            frame,
            config.calibration_fraction,
            config.split_seed + domain_id,
        )
        fit_frames.append(_subset(frame, fit_units))
        calibration_frames.append(_subset(frame, calibration_units))
        fit_engine_ids.extend((domain_id * 10_000 + fit_units).tolist())
        calibration_engine_ids.extend((domain_id * 10_000 + calibration_units).tolist())

    features = sensor_columns()
    statistics = fit_source_statistics(
        fit_frames,
        features,
        preprocessing="global_zscore",
        seed=config.split_seed,
    )
    fit_parts: list[tuple[np.ndarray, ...]] = []
    calibration_parts: list[tuple[np.ndarray, ...]] = []
    for domain_id, (fit_frame, calibration_frame) in enumerate(
        zip(fit_frames, calibration_frames)
    ):
        fit_normalized = normalize_settings(normalize(fit_frame, statistics), statistics)
        calibration_normalized = normalize_settings(
            normalize(calibration_frame, statistics), statistics
        )
        fit_parts.append(
            make_source_windows(fit_normalized, features, config.window_size, domain_id)
        )
        calibration_parts.append(
            make_source_windows(
                calibration_normalized, features, config.window_size, domain_id
            )
        )
    return PreparedMDMTSources(
        fit=_combine(fit_parts),
        calibration=_combine(calibration_parts),
        statistics=statistics,
        source_domains=source_domains,
        fit_engine_ids=tuple(sorted(fit_engine_ids)),
        calibration_engine_ids=tuple(sorted(calibration_engine_ids)),
    )


@torch.no_grad()
def _predict(
    model: MDMTBiLSTM,
    windows: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, float]:
    torch_device = torch.device(device)
    model.eval().to(torch_device)
    predictions: list[np.ndarray] = []
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    started = time.perf_counter()
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start : start + batch_size]).to(
            torch_device, non_blocking=torch_device.type == "cuda"
        )
        predictions.append(model(batch)["prediction"].cpu().numpy())
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - started
    return np.concatenate(predictions), elapsed * 1000.0 / max(1, len(windows))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.square(y_true - y_true.mean()).sum())
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - np.square(y_true - y_pred).sum() / denominator)


def _worst_decile_rmse(errors: np.ndarray) -> float:
    count = max(1, int(np.ceil(0.1 * len(errors))))
    return float(np.sqrt(np.sort(np.square(errors))[-count:].mean()))


def execute(
    config: ProtocolV1Config,
    force_train: bool = False,
    prepared_sources: PreparedMDMTSources | None = None,
    artifact_output_dir: Path | None = None,
) -> dict:
    run_dir = config.output_dir / "mdmt" / config.target_domain / f"seed_{config.seed}"
    artifact_root = artifact_output_dir or config.output_dir
    artifact_run_dir = (
        artifact_root / "mdmt" / config.target_domain / f"seed_{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    access_sequence = ["source_files_opened"]
    prepared = prepared_sources or prepare_mdmt_sources(config)
    if prepared.source_domains != tuple(
        domain for domain in DOMAINS if domain != config.target_domain
    ):
        raise ValueError("Prepared source domains do not match the held-out target.")
    _save_statistics(run_dir / "preprocessing_stats.npz", prepared.statistics)
    _save_json(
        run_dir / "registered_config.json",
        {
            "protocol": PROTOCOL_NAME,
            "config": _registered_config(config, artifact_root),
            "source_domains": prepared.source_domains,
            "sensors": list(sensor_columns()),
            "stage_schedule": mdmt_stage_schedule(tuple(np.unique(prepared.fit.domain))),
            "batch_size_per_domain": MDMT_BATCH_SIZE,
            "temperature": MDMT_TEMPERATURE,
            "perturbation": MDMT_PERTURBATION,
            "source_loss_weight": MDMT_SOURCE_WEIGHT,
            "kernel_multipliers": MDMT_KERNEL_MULTIPLIERS,
        },
    )

    checkpoint = run_dir / "checkpoints" / "mdmt_final.pt"
    started = time.perf_counter()
    prior_audit_path = run_dir / "mdmt_audit.json"
    prior_training_seconds: float | None = None
    if prior_audit_path.exists():
        prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
        if prior_audit.get("protocol") == PROTOCOL_NAME:
            prior_training_seconds = float(
                prior_audit.get(
                    "training_seconds",
                    prior_audit.get("training_or_load_seconds", 0.0),
                )
            )
    if checkpoint.exists() and not force_train:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol") != PROTOCOL_NAME:
            raise RuntimeError("Existing MDMT checkpoint uses a different protocol.")
        model = MDMTBiLSTM(sensors=prepared.fit.windows.shape[2])
        model.load_state_dict(payload["model_state"])
        history = payload["history"]
        access_sequence.append("source_checkpoint_loaded")
    else:
        model, history = train_mdmt(prepared.fit, config.seed, config.device)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol": PROTOCOL_NAME,
                "target_domain": config.target_domain,
                "source_domains": prepared.source_domains,
                "seed": config.seed,
                "model_state": model.state_dict(),
                "history": history,
                "parameter_count": trainable_parameter_count(model),
                "paper_parameter_count": PAPER_PARAMETER_COUNT,
            },
            checkpoint,
        )
        access_sequence.append("source_training_completed_and_checkpoint_saved")
    training_or_load_seconds = time.perf_counter() - started
    training_seconds = (
        prior_training_seconds
        if checkpoint.exists() and not force_train and prior_training_seconds is not None
        else training_or_load_seconds
    )
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    parameter_count = trainable_parameter_count(model)
    if parameter_count != REPRODUCED_PARAMETER_COUNT:
        raise RuntimeError("MDMT parameter-count audit failed after training.")
    calibration_prediction, _ = _predict(
        model,
        prepared.calibration.windows,
        config.tta_inference_batch_size,
        config.device,
    )
    calibration_metrics = {
        **metric_dict(prepared.calibration.rul, calibration_prediction),
        "r2": _r2(prepared.calibration.rul, calibration_prediction),
    }
    _save_json(run_dir / "source_calibration_metrics.json", calibration_metrics)
    access_sequence.append("source_calibration_diagnostic_saved_without_selection")

    # The first target access occurs only after source training/checkpointing.
    target_frame = load_test_domain(config.raw_dir, config.target_domain)
    target_frame = normalize(target_frame, prepared.statistics)
    target_frame = normalize_settings(target_frame, prepared.statistics)
    target = make_target_stream(
        target_frame,
        sensor_columns(),
        config.window_size,
    )
    access_sequence.append("unlabeled_target_covariates_opened_after_training")
    predictions, latency = _predict(
        model,
        target.windows,
        config.tta_inference_batch_size,
        config.device,
    )
    prediction_table = pd.DataFrame(
        {
            "engine_id": target.engine,
            "cycle": target.cycle,
            "is_final": target.is_final,
            "prediction": predictions,
        }
    )
    prediction_table.to_csv(run_dir / "predictions_before_labels.csv", index=False)
    access_sequence.append("target_predictions_saved_before_labels")

    final = (
        prediction_table.loc[prediction_table["is_final"]]
        .sort_values("engine_id")
        .reset_index(drop=True)
    )
    target_rul = load_target_rul(
        config.raw_dir,
        config.target_domain,
        expected_engines=len(final),
    )
    access_sequence.append("official_target_rul_opened_for_scoring")
    final["true_rul"] = target_rul
    final["error"] = final["prediction"] - final["true_rul"]
    final.to_csv(run_dir / "engine_predictions.csv", index=False)
    true = final["true_rul"].to_numpy(dtype=np.float64)
    predicted = final["prediction"].to_numpy(dtype=np.float64)
    errors = predicted - true
    metrics = {
        **metric_dict(true, predicted),
        "r2": _r2(true, predicted),
        "worst_10pct_engine_rmse": _worst_decile_rmse(errors),
        "window_latency_ms": latency,
        "parameter_count": parameter_count,
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("MDMT final metrics contain NaN or Inf.")
    _save_json(run_dir / "metrics.json", metrics)

    audit = {
        "protocol": PROTOCOL_NAME,
        "registered_run": True,
        "target_domain": config.target_domain,
        "source_domains": prepared.source_domains,
        "seed": config.seed,
        "preprocessing": "global_zscore_source_fit_only",
        "window_size": config.window_size,
        "stride": 1,
        "rul_cap": config.rul_cap,
        "selected_sensors": list(sensor_columns()),
        "target_covariates_loaded_before_training": False,
        "target_windows_used_during_training": False,
        "target_labels_used_during_training_or_selection": False,
        "target_predictions_saved_before_labels": True,
        "target_time_updates": 0,
        "checkpoint_policy": "fixed_final_stage_no_metric_selection",
        "calibration_used_for_checkpoint_selection": False,
        "source_fit_engines": list(prepared.fit_engine_ids),
        "source_calibration_engines": list(prepared.calibration_engine_ids),
        "partitions_disjoint": not bool(
            set(prepared.fit_engine_ids) & set(prepared.calibration_engine_ids)
        ),
        "source_fit_windows": int(len(prepared.fit.rul)),
        "source_calibration_windows": int(len(prepared.calibration.rul)),
        "parameter_count": parameter_count,
        "paper_parameter_count": PAPER_PARAMETER_COUNT,
        "parameter_count_difference": parameter_count - PAPER_PARAMETER_COUNT,
        "stage_schedule": mdmt_stage_schedule(tuple(np.unique(prepared.fit.domain))),
        "batch_size_per_domain": MDMT_BATCH_SIZE,
        "temperature": MDMT_TEMPERATURE,
        "perturbation": MDMT_PERTURBATION,
        "source_loss_weight": MDMT_SOURCE_WEIGHT,
        "kernel_multipliers": MDMT_KERNEL_MULTIPLIERS,
        "access_sequence": access_sequence,
        "training_seconds": training_seconds,
        "checkpoint_load_or_training_seconds": training_or_load_seconds,
        "training_or_load_seconds": training_seconds,
        "checkpoint": str(
            (artifact_run_dir / "checkpoints" / "mdmt_final.pt").resolve()
        ),
        "source_calibration_metrics": calibration_metrics,
        "metrics": metrics,
        "config": _registered_config(config, artifact_root),
    }
    _save_json(run_dir / "mdmt_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=REPORT_SEEDS,
        default=list(REPORT_SEEDS),
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_CMAPSS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKBENCH_RESULTS_DIR / "mdmt_unified_protocol_v1",
    )
    parser.add_argument(
        "--artifact-output-dir",
        type=Path,
        default=None,
        help="Final registered location when execution uses a temporary staging root.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    artifact_output_dir = (
        args.artifact_output_dir.resolve()
        if args.artifact_output_dir is not None
        else None
    )
    for target in args.targets:
        source_config = ProtocolV1Config(
            target_domain=target,
            raw_dir=raw_dir,
            output_dir=output_dir,
            seed=int(args.seeds[0]),
            device=args.device,
        )
        prepared = prepare_mdmt_sources(source_config)
        for seed in args.seeds:
            config = ProtocolV1Config(
                target_domain=target,
                raw_dir=raw_dir,
                output_dir=output_dir,
                seed=int(seed),
                device=args.device,
            )
            audit = execute(
                config,
                args.force_train,
                prepared,
                artifact_output_dir=artifact_output_dir,
            )
            print(
                json.dumps(
                    {
                        "target": target,
                        "seed": seed,
                        "rmse": audit["metrics"]["rmse"],
                        "mae": audit["metrics"]["mae"],
                        "r2": audit["metrics"]["r2"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
