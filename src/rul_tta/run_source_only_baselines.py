from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import load_target_rul
from .metrics import metric_dict
from .ms_dsn_baseline import MSDSNSourceOnlyRULModel, ms_dsn_loss
from .paths import CURRENT_RESULTS_DIR, DEFAULT_CMAPSS_DIR
from .protocol_v1_config import (
    DOMAINS,
    FROZEN_FIXED_EPOCHS,
    REPORT_SEEDS,
    ProtocolV1Config,
)
from .protocol_v1_data import prepare_protocol_fold
from .protocol_v1_models import build_conditioncomp_rul_net
from .protocol_v1_train import domain_balanced_loader
from .run_p3_mechanism_ablation import _predict_target, _save_json


BASELINES = ("slow_coral", "slow_group_dro", "slow_vrex", "slow_ms_dsn")
CORAL_WEIGHT = 0.1
GROUP_DRO_ETA = 0.01
GROUP_DRO_MSE_SCALE = 125.0**2
VREX_WEIGHT = 1.0
MS_DSN_PAPER_DOI = "10.1109/TCYB.2025.3618124"
MS_DSN_BATCH_SIZE = 64
MS_DSN_IMPLEMENTATION = "same-condition-residual-backbone-reimplementation-v6"
MS_DSN_TASK_WEIGHT = 1000.0


def _coral_penalty(
    features: torch.Tensor,
    domains: torch.Tensor,
) -> torch.Tensor:
    covariance: list[torch.Tensor] = []
    for domain in torch.unique(domains):
        selected = features[domains == domain]
        if len(selected) < 2:
            continue
        centered = selected - selected.mean(dim=0, keepdim=True)
        covariance.append(centered.T @ centered / (len(selected) - 1))
    penalty = torch.zeros((), device=features.device)
    pairs = 0
    for left in range(len(covariance)):
        for right in range(left + 1, len(covariance)):
            penalty = penalty + (
                covariance[left] - covariance[right]
            ).square().mean()
            pairs += 1
    return penalty / max(1, pairs)


def _baseline_loss(
    baseline: str,
    squared_error: torch.Tensor,
    features: torch.Tensor,
    domains: torch.Tensor,
    group_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if baseline == "slow_coral":
        penalty = _coral_penalty(features, domains)
        loss = squared_error.mean() + CORAL_WEIGHT * penalty
        return loss, {"coral": float(penalty.detach())}
    if baseline == "slow_group_dro":
        present_domains = torch.unique(domains, sorted=True)
        group_losses = torch.stack(
            [
                squared_error[domains == domain].mean()
                for domain in present_domains
            ]
        )
        if group_weights is None:
            active_weights = torch.ones_like(group_losses)
            active_weights /= active_weights.sum()
        else:
            with torch.no_grad():
                logits = torch.log(group_weights.clamp_min(1e-12))
                logits[present_domains] += (
                    GROUP_DRO_ETA
                    * group_losses.detach()
                    / GROUP_DRO_MSE_SCALE
                )
                group_weights.copy_(torch.softmax(logits, dim=0))
            active_weights = group_weights[present_domains]
            active_weights = active_weights / active_weights.sum()
        loss = torch.sum(active_weights * group_losses)
        return loss, {
            "worst_group_mse": float(group_losses.max().detach()),
            "max_group_weight": float(active_weights.max().detach()),
        }
    if baseline == "slow_vrex":
        present_domains = torch.unique(domains, sorted=True)
        group_losses = torch.stack(
            [
                squared_error[domains == domain].mean()
                for domain in present_domains
            ]
        )
        normalized = group_losses / GROUP_DRO_MSE_SCALE
        penalty = normalized.var(unbiased=False) * GROUP_DRO_MSE_SCALE
        loss = group_losses.mean() + VREX_WEIGHT * penalty
        return loss, {
            "vrex": float(penalty.detach()),
            "worst_group_mse": float(group_losses.max().detach()),
        }
    raise ValueError(f"Unknown baseline: {baseline}")


def _train(
    source,
    config: ProtocolV1Config,
    baseline: str,
    ms_dsn_task_weight: float = MS_DSN_TASK_WEIGHT,
    representation_mode: str = "raw_only",
) -> tuple[
    nn.Module,
    list[dict[str, float]],
    list[float],
]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    group_count = int(np.max(source.domain)) + 1
    if baseline == "slow_ms_dsn":
        model: nn.Module = MSDSNSourceOnlyRULModel(
            sensors=source.windows.shape[2],
            window_size=config.window_size,
            slow_bins=config.slow_bins,
            source_domains=group_count,
            rul_scale=config.rul_cap,
            representation_mode=representation_mode,
        ).to(device)
    else:
        model = build_conditioncomp_rul_net(
            sensors=source.windows.shape[2],
            window_size=config.window_size,
            retained_bins=config.slow_bins,
            representation_mode=representation_mode,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    training_batch_size = (
        MS_DSN_BATCH_SIZE if baseline == "slow_ms_dsn" else config.batch_size
    )
    loader = domain_balanced_loader(
        source,
        training_batch_size,
        config.seed,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    history: list[dict[str, float]] = []
    group_weights = torch.full(
        (group_count,),
        1.0 / group_count,
        device=device,
    )
    for epoch in range(1, FROZEN_FIXED_EPOCHS + 1):
        model.train()
        total = 0.0
        samples = 0
        auxiliary_total = 0.0
        for batch in loader:
            non_blocking = device.type == "cuda"
            windows = batch["window"].to(device, non_blocking=non_blocking)
            target = batch["rul"].to(device, non_blocking=non_blocking)
            domains = batch["domain"].to(device, non_blocking=non_blocking)
            if baseline == "slow_ms_dsn":
                output = model(windows, domains)
                loss, diagnostics = ms_dsn_loss(
                    output,
                    target,
                    domains,
                    rul_scale=config.rul_cap,
                    task_weight=ms_dsn_task_weight,
                )
            else:
                output = model(windows)
                squared_error = (output["prediction"] - target).square()
                loss, diagnostics = _baseline_loss(
                    baseline,
                    squared_error,
                    output["fused_features"],
                    domains,
                    group_weights if baseline == "slow_group_dro" else None,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(target)
            auxiliary_total += sum(diagnostics.values()) * len(target)
            samples += len(target)
        history.append(
            {
                "epoch": float(epoch),
                "objective": total / max(1, samples),
                "auxiliary": auxiliary_total / max(1, samples),
            }
        )
    return model, history, group_weights.detach().cpu().tolist()


def execute(
    config: ProtocolV1Config,
    baseline: str,
    force_train: bool = False,
    prepared_fold=None,
    representation_mode: str = "raw_only",
) -> dict:
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}")
    if representation_mode not in {"slow_only", "raw_only"}:
        raise ValueError("representation_mode must be slow_only or raw_only")
    fold = (
        prepared_fold
        if prepared_fold is not None
        else prepare_protocol_fold(config, preprocessing="global_zscore")
    )
    if fold.statistics.preprocessing != "global_zscore":
        raise ValueError("Literature baselines require global_zscore.")
    run_dir = (
        config.output_dir
        / baseline
        / config.target_domain
        / f"seed_{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoints" / "source_model.pt"
    started = time.perf_counter()
    if checkpoint.exists() and not force_train:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "baseline": baseline,
            "target_domain": config.target_domain,
            "seed": config.seed,
            "epochs": FROZEN_FIXED_EPOCHS,
            "representation_mode": representation_mode,
        }
        observed = {
            "baseline": payload.get("baseline"),
            "target_domain": payload.get("target_domain"),
            "seed": payload.get("seed"),
            "epochs": payload.get("epochs"),
            "representation_mode": payload.get("representation_mode"),
        }
        if observed != expected:
            raise ValueError(f"Baseline checkpoint mismatch: {observed}")
        if baseline == "slow_ms_dsn":
            model = MSDSNSourceOnlyRULModel(
                sensors=fold.fit.windows.shape[2],
                window_size=config.window_size,
                slow_bins=config.slow_bins,
                source_domains=int(np.max(fold.fit.domain)) + 1,
                rul_scale=config.rul_cap,
                representation_mode=representation_mode,
            )
        else:
            model = build_conditioncomp_rul_net(
                sensors=fold.fit.windows.shape[2],
                window_size=config.window_size,
                retained_bins=config.slow_bins,
                representation_mode=representation_mode,
            )
        model.load_state_dict(payload["model_state"])
        history = payload["history"]
        final_group_weights = payload.get("final_group_weights")
    else:
        model, history, final_group_weights = _train(
            fold.fit,
            config,
            baseline,
            MS_DSN_TASK_WEIGHT,
            representation_mode,
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol": "source-only-literature-baselines-v2",
                "baseline": baseline,
                "target_domain": config.target_domain,
                "seed": config.seed,
                "epochs": FROZEN_FIXED_EPOCHS,
                "preprocessing": "global_zscore",
                "representation_mode": representation_mode,
                "coral_weight": CORAL_WEIGHT,
                "group_dro_eta": GROUP_DRO_ETA,
                "group_dro_mse_scale": GROUP_DRO_MSE_SCALE,
                "vrex_weight": VREX_WEIGHT,
                "ms_dsn_paper_doi": MS_DSN_PAPER_DOI,
                "ms_dsn_implementation": MS_DSN_IMPLEMENTATION,
                "ms_dsn_task_weight": (
                    MS_DSN_TASK_WEIGHT if baseline == "slow_ms_dsn" else None
                ),
                "training_batch_size": (
                    MS_DSN_BATCH_SIZE
                    if baseline == "slow_ms_dsn"
                    else config.batch_size
                ),
                "final_group_weights": final_group_weights,
                "model_state": model.state_dict(),
                "history": history,
            },
            checkpoint,
        )
    training_or_load_seconds = time.perf_counter() - started

    predictions, latency = _predict_target(
        model,
        fold.target.windows,
        config.tta_inference_batch_size,
        config.device,
    )
    table = pd.DataFrame(
        {
            "engine_id": fold.target.engine,
            "cycle": fold.target.cycle,
            "is_final": fold.target.is_final,
            "prediction": predictions,
        }
    )
    table.to_csv(run_dir / "predictions_before_labels.csv", index=False)
    final = (
        table.loc[table["is_final"]]
        .sort_values("engine_id")
        .reset_index(drop=True)
    )
    target_rul = load_target_rul(
        config.raw_dir,
        config.target_domain,
        expected_engines=len(final),
    )
    final["true_rul"] = target_rul
    final["error"] = final["prediction"] - final["true_rul"]
    final.to_csv(run_dir / "engine_predictions.csv", index=False)
    errors = final["error"].to_numpy(dtype=np.float64)
    metrics = {
        **metric_dict(
            final["true_rul"].to_numpy(dtype=np.float64),
            final["prediction"].to_numpy(dtype=np.float64),
        ),
        "worst_10pct_engine_rmse": float(
            np.sqrt(
                np.mean(
                    np.sort(np.square(errors))[
                        -max(1, int(np.ceil(0.1 * len(errors)))) :
                    ]
                )
            )
        ),
        "window_latency_ms": latency,
    }
    _save_json(run_dir / "metrics.json", metrics)
    audit = {
        "protocol": "source-only-literature-baselines-v2",
        "registered_run": True,
        "baseline": baseline,
        "target_domain": config.target_domain,
        "seed": config.seed,
        "epochs": FROZEN_FIXED_EPOCHS,
        "preprocessing": "global_zscore",
        "representation_mode": representation_mode,
        "target_windows_used_during_training": False,
        "target_predictions_saved_before_labels": True,
        "target_time_updates": 0,
        "source_fit_engines": list(fold.fit_engine_ids),
        "source_calibration_engines": list(fold.calibration_engine_ids),
        "partitions_disjoint": not bool(
            set(fold.fit_engine_ids) & set(fold.calibration_engine_ids)
        ),
        "coral_weight": CORAL_WEIGHT,
        "group_dro_eta": GROUP_DRO_ETA,
        "group_dro_mse_scale": GROUP_DRO_MSE_SCALE,
        "vrex_weight": VREX_WEIGHT,
        "ms_dsn_paper_doi": MS_DSN_PAPER_DOI if baseline == "slow_ms_dsn" else None,
        "ms_dsn_implementation": (
            MS_DSN_IMPLEMENTATION if baseline == "slow_ms_dsn" else None
        ),
        "ms_dsn_task_weight": (
            MS_DSN_TASK_WEIGHT if baseline == "slow_ms_dsn" else None
        ),
        "training_batch_size": (
            MS_DSN_BATCH_SIZE if baseline == "slow_ms_dsn" else config.batch_size
        ),
        "final_group_weights": final_group_weights,
        "training_or_load_seconds": training_or_load_seconds,
        "checkpoint": str(checkpoint.resolve()),
        "config": {
            **asdict(config),
            "raw_dir": str(config.raw_dir),
            "output_dir": str(config.output_dir),
        },
        "metrics": metrics,
    }
    _save_json(run_dir / "baseline_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=DOMAINS)
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--seed", required=True, type=int, choices=REPORT_SEEDS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_CMAPSS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_RESULTS_DIR / "raw_backbone_baselines_v1",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--representation-mode",
        choices=("slow_only", "raw_only"),
        default="raw_only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProtocolV1Config(
        target_domain=args.target,
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
        device=args.device,
        epochs=FROZEN_FIXED_EPOCHS,
    )
    result = execute(
        config,
        args.baseline,
        args.force_train,
        representation_mode=args.representation_mode,
    )
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
