from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_target_rul
from .metrics import metric_dict
from .paths import CURRENT_RESULTS_DIR, DEFAULT_CMAPSS_DIR
from .protocol_v1_config import (
    DOMAINS,
    FROZEN_FIXED_EPOCHS,
    PROTOCOL_V1_NAME,
    PROTOCOL_V1_VERSION,
    REPORT_SEEDS,
    ProtocolV1Config,
)
from .protocol_v1_data import prepare_protocol_fold
from .protocol_v1_models import ConditionCompRULNet, build_conditioncomp_rul_net
from .protocol_v1_train import train_final_p3


FACTORIAL_ABLATION_VARIANTS = (
    "raw_no_condition",
    "raw_compensated",
    "slow_no_condition",
    "slow_only",
)
ABLATION_VARIANTS = FACTORIAL_ABLATION_VARIANTS
FORMAL_ABLATION_VARIANTS = FACTORIAL_ABLATION_VARIANTS
FINAL_SLOW_ABLATION_VARIANTS = ("slow_no_condition", "slow_only")


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


def _save_statistics(path: Path, statistics) -> None:
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


def _variant_settings(variant: str) -> tuple[str, str]:
    if variant == "slow_only":
        return "continuous_comp", "slow_only"
    if variant == "slow_no_condition":
        return "global_zscore", "slow_only"
    if variant == "raw_no_condition":
        return "global_zscore", "raw_only"
    if variant == "raw_compensated":
        return "continuous_comp", "raw_only"
    raise ValueError(f"Unknown ablation variant: {variant}")


def _load_checkpoint(
    path: Path,
    config: ProtocolV1Config,
    sensors: int,
    variant: str,
    representation_mode: str,
) -> ConditionCompRULNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "protocol_version": PROTOCOL_V1_VERSION,
        "ablation_variant": variant,
        "representation_mode": representation_mode,
        "target_domain": config.target_domain,
        "seed": config.seed,
        "fixed_epochs": FROZEN_FIXED_EPOCHS,
    }
    observed = {
        "protocol_version": payload.get("protocol_version"),
        "ablation_variant": payload.get("ablation_variant"),
        "representation_mode": payload.get("representation_mode"),
        "target_domain": payload.get("config", {}).get("target_domain"),
        "seed": payload.get("config", {}).get("seed"),
        "fixed_epochs": payload.get("selected_epoch"),
    }
    mismatches = {
        key: (observed[key], value)
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        raise ValueError(f"Ablation checkpoint mismatch: {mismatches}")
    model = build_conditioncomp_rul_net(
        sensors=sensors,
        window_size=config.window_size,
        retained_bins=config.slow_bins,
        representation_mode=representation_mode,
    )
    model.load_state_dict(payload["model_state"])
    return model


@torch.no_grad()
def _predict_target(
    model: ConditionCompRULNet,
    windows: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, float]:
    torch_device = torch.device(device)
    model.eval().to(torch_device)
    predicted: list[np.ndarray] = []
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    started = time.perf_counter()
    for start in range(0, len(windows), batch_size):
        values = torch.from_numpy(windows[start : start + batch_size]).to(
            torch_device, non_blocking=torch_device.type == "cuda"
        )
        predicted.append(model(values)["prediction"].cpu().numpy())
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - started
    return np.concatenate(predicted), elapsed * 1000.0 / max(1, len(windows))


def execute(
    config: ProtocolV1Config,
    variant: str,
    fixed_epochs: int,
    force_train: bool = False,
    prepared_fold=None,
) -> dict:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"variant must be one of {ABLATION_VARIANTS}")
    if fixed_epochs != FROZEN_FIXED_EPOCHS:
        raise ValueError(
            f"Registered ablations require exactly {FROZEN_FIXED_EPOCHS} epochs."
        )
    preprocessing, representation_mode = _variant_settings(variant)
    run_dir = (
        config.output_dir
        / variant
        / config.target_domain
        / f"seed_{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    fold = (
        prepared_fold
        if prepared_fold is not None
        else prepare_protocol_fold(config, preprocessing=preprocessing)
    )
    if fold.statistics.preprocessing != preprocessing:
        raise ValueError(
            "Prepared fold preprocessing does not match the ablation variant."
        )
    _save_statistics(run_dir / "preprocessing.npz", fold.statistics)

    checkpoint = run_dir / "checkpoints" / "p3_ablation.pt"
    if checkpoint.exists() and not force_train:
        model = _load_checkpoint(
            checkpoint,
            config,
            fold.fit.windows.shape[2],
            variant,
            representation_mode,
        )
    else:
        result = train_final_p3(
            fold.fit,
            config,
            fixed_epochs,
            checkpoint,
            representation_mode=representation_mode,
        )
        model = result.model
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload.update(
            {
                "method": "P3-mechanism-ablation",
                "ablation_variant": variant,
                "preprocessing": preprocessing,
                "representation_mode": representation_mode,
            }
        )
        torch.save(payload, checkpoint)

    predictions, latency = _predict_target(
        model,
        fold.target.windows,
        config.tta_inference_batch_size,
        config.device,
    )
    prediction_table = pd.DataFrame(
        {
            "engine_id": fold.target.engine,
            "cycle": fold.target.cycle,
            "is_final": fold.target.is_final,
            "prediction": predictions,
        }
    )
    prediction_table.to_csv(
        run_dir / "predictions_before_labels.csv", index=False
    )
    final = (
        prediction_table.loc[prediction_table["is_final"]]
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    # The target-label file is deliberately opened only after all predictions
    # have been materialized on disk.
    target_rul = load_target_rul(
        config.raw_dir,
        config.target_domain,
        expected_engines=len(final),
    )
    expected_engines = np.arange(1, len(target_rul) + 1, dtype=np.int64)
    if not np.array_equal(
        final["engine_id"].to_numpy(dtype=np.int64), expected_engines
    ):
        raise ValueError("Target engine IDs do not align with official RUL rows.")
    final["true_rul"] = target_rul
    final["error"] = final["prediction"] - final["true_rul"]
    final["variant"] = variant
    final.to_csv(run_dir / "engine_predictions.csv", index=False)

    metrics = {
        **metric_dict(
            final["true_rul"].to_numpy(dtype=np.float64),
            final["prediction"].to_numpy(dtype=np.float64),
        ),
        "worst_10pct_engine_rmse": float(
            np.sqrt(
                np.mean(
                    np.sort(np.square(final["error"].to_numpy(dtype=np.float64)))[
                        -max(1, int(np.ceil(0.1 * len(final)))) :
                    ]
                )
            )
        ),
        "window_latency_ms": latency,
    }
    _save_json(run_dir / "metrics.json", metrics)
    audit = {
        "protocol": PROTOCOL_V1_NAME,
        "protocol_version": PROTOCOL_V1_VERSION,
        "registered_mechanism_ablation": (
            config.seed in REPORT_SEEDS
            and config.target_domain in DOMAINS
            and fixed_epochs == FROZEN_FIXED_EPOCHS
            and variant in FORMAL_ABLATION_VARIANTS
        ),
        "registered_final_slow_ablation": (
            config.seed in REPORT_SEEDS
            and config.target_domain in DOMAINS
            and fixed_epochs == FROZEN_FIXED_EPOCHS
            and variant in FINAL_SLOW_ABLATION_VARIANTS
        ),
        "registered_factorial_ablation": (
            config.seed in REPORT_SEEDS
            and config.target_domain in DOMAINS
            and fixed_epochs == FROZEN_FIXED_EPOCHS
            and variant in FACTORIAL_ABLATION_VARIANTS
        ),
        "variant": variant,
        "preprocessing": preprocessing,
        "representation_mode": representation_mode,
        "target_domain": config.target_domain,
        "seed": config.seed,
        "fixed_epochs": fixed_epochs,
        "source_fit_engines": list(fold.fit_engine_ids),
        "source_calibration_engines": list(fold.calibration_engine_ids),
        "partitions_disjoint": not bool(
            set(fold.fit_engine_ids) & set(fold.calibration_engine_ids)
        ),
        "target_predictions_saved_before_labels": True,
        "target_time_updates": 0,
        "fast_adapter_requested": False,
        "checkpoint": str(checkpoint.resolve()),
        "config": {
            **asdict(config),
            "raw_dir": str(config.raw_dir),
            "output_dir": str(config.output_dir),
        },
        "metrics": metrics,
    }
    _save_json(run_dir / "ablation_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P3 mechanism ablation")
    parser.add_argument("--target", required=True, choices=DOMAINS)
    parser.add_argument("--variant", required=True, choices=ABLATION_VARIANTS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_CMAPSS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_RESULTS_DIR / "condition_slow_factorial_v1",
    )
    parser.add_argument("--seed", type=int, required=True, choices=REPORT_SEEDS)
    parser.add_argument(
        "--fixed-epochs", type=int, default=FROZEN_FIXED_EPOCHS
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProtocolV1Config(
        target_domain=args.target,
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
        device=args.device,
        epochs=args.fixed_epochs,
    )
    audit = execute(config, args.variant, args.fixed_epochs, args.force_train)
    print(json.dumps(audit["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
