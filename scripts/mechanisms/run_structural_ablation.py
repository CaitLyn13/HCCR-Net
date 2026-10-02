"""Four-fold, five-seed HCCR-Net structural ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/structural_ablation"
VARIANTS = ("dual_branch", "temporal_only", "sensor_only", "single_receptive_field")

from rul_tta.data import load_target_rul
from rul_tta.metrics import metric_dict
from rul_tta.protocol_v1_config import (
    DOMAINS,
    FROZEN_FIXED_EPOCHS,
    REPORT_SEEDS,
    ProtocolV1Config,
)
from rul_tta.protocol_v1_data import prepare_protocol_fold
from rul_tta.protocol_v1_models import (
    CausalConvSummary,
    ConditionCompRULNet,
)
from rul_tta.protocol_v1_train import _train_epochs
from rul_tta.run_p3_mechanism_ablation import _predict_target


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StructuralVariant(ConditionCompRULNet):
    """Keep the registered head and 256-D fusion input; mask an omitted branch."""

    def __init__(self, variant: str, sensors: int, window_size: int):
        super().__init__(sensors=sensors, window_size=window_size)
        if variant not in VARIANTS[1:]:
            raise ValueError(variant)
        self.variant = variant
        if variant == "single_receptive_field":
            # One 128-channel causal convolution retains temporal output width.
            self.temporal_encoder = CausalConvSummary(sensors, 128, kernel_size=9)

    def encode(self, residual: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.variant == "temporal_only":
            temporal = self.temporal_encoder(residual)
            sensor = temporal.new_zeros((len(residual), 128))
        elif self.variant == "sensor_only":
            sensor = self.sensor_encoder(residual)
            temporal = sensor.new_zeros((len(residual), 128))
        else:
            temporal = self.temporal_encoder(residual)
            sensor = self.sensor_encoder(residual)
        fused = self.fusion(torch.cat([temporal, sensor], dim=1))
        return {
            "model_input": residual,
            "temporal_features": temporal,
            "sensor_features": sensor,
            "fused_features": fused,
        }


def build_model(variant: str, sensors: int, window_size: int) -> ConditionCompRULNet:
    if variant == "dual_branch":
        return ConditionCompRULNet(sensors=sensors, window_size=window_size)
    return StructuralVariant(variant, sensors=sensors, window_size=window_size)


def train_and_score(fold, config: ProtocolV1Config, variant: str, run_dir: Path) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = build_model(variant, fold.fit.windows.shape[2], config.window_size)
    started = time.perf_counter()
    _train_epochs(model, fold.fit, config, FROZEN_FIXED_EPOCHS, validation=None)
    train_seconds = time.perf_counter() - started
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant,
            "target": config.target_domain,
            "seed": config.seed,
            "selected_epoch": FROZEN_FIXED_EPOCHS,
            "model_state": model.state_dict(),
            "config": {**asdict(config), "raw_dir": str(config.raw_dir), "output_dir": str(config.output_dir)},
        },
        run_dir / "checkpoint.pt",
    )
    prediction, latency = _predict_target(
        model, fold.target.windows, config.tta_inference_batch_size, config.device
    )
    rows = pd.DataFrame(
        {
            "engine_id": fold.target.engine,
            "cycle": fold.target.cycle,
            "is_final": fold.target.is_final,
            "prediction": prediction,
        }
    )
    rows.to_csv(run_dir / "predictions_before_labels.csv", index=False)
    final = rows.loc[rows.is_final].sort_values("engine_id").reset_index(drop=True)
    true_rul = load_target_rul(config.raw_dir, config.target_domain, expected_engines=len(final))
    if not np.array_equal(final.engine_id.to_numpy(), np.arange(1, len(true_rul) + 1)):
        raise RuntimeError(f"Target engine ID mismatch: {variant}/{config.target_domain}/{config.seed}")
    final["true_rul"] = true_rul
    final["error"] = final.prediction - final.true_rul
    final.to_csv(run_dir / "engine_predictions.csv", index=False)
    score = metric_dict(final.true_rul.to_numpy(np.float64), final.prediction.to_numpy(np.float64))
    score["worst_10pct_engine_rmse"] = float(
        np.sqrt(np.mean(np.sort(np.square(final.error.to_numpy(np.float64)))[-max(1, int(np.ceil(0.1 * len(final)))) :]))
    )
    audit = {
        "variant": variant,
        "target": config.target_domain,
        "seed": config.seed,
        "source_only": True,
        "preprocessing": "continuous_comp",
        "fixed_epochs": FROZEN_FIXED_EPOCHS,
        "training_seconds": train_seconds,
        "window_latency_ms": latency,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "source_fit_engines": list(fold.fit_engine_ids),
        "source_calibration_engines": list(fold.calibration_engine_ids),
        "checkpoint": str((run_dir / "checkpoint.pt").resolve()),
        "metrics": score,
    }
    (run_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def aggregate(output: Path) -> None:
    records = []
    for target in DOMAINS:
        for seed in REPORT_SEEDS:
            for variant in VARIANTS:
                path = output / variant / target / f"seed_{seed}" / "audit.json"
                if not path.is_file():
                    return
                audit = json.loads(path.read_text(encoding="utf-8"))
                records.append({"variant": variant, "target": target, "seed": seed, **audit["metrics"]})
    runs = pd.DataFrame(records)
    runs.to_csv(output / "runs.csv", index=False)
    metrics = ["rmse", "mae", "phm08", "signed_bias", "worst_10pct_engine_rmse"]
    by_target = runs.groupby(["variant", "target"])[metrics].agg(["mean", "std"])
    by_target.to_csv(output / "by_target.csv")
    macro = runs.groupby(["variant", "target"])[metrics].mean().groupby("variant").mean()
    macro.to_csv(output / "macro.csv")
    baseline = runs.loc[runs.variant == "dual_branch", ["target", "seed", "rmse", "phm08"]]
    comparisons = []
    for variant in VARIANTS[1:]:
        variant_runs = runs.loc[runs.variant == variant, ["target", "seed", "rmse", "phm08"]]
        paired = variant_runs.merge(baseline, on=["target", "seed"], suffixes=("_variant", "_dual"), validate="one_to_one")
        paired["rmse_difference"] = paired.rmse_variant - paired.rmse_dual
        paired["phm08_difference"] = paired.phm08_variant - paired.phm08_dual
        paired.insert(0, "variant", variant)
        comparisons.append(paired)
    pd.concat(comparisons, ignore_index=True).to_csv(output / "paired_vs_dual.csv", index=False)
    print("AGGREGATE_COMPLETE", flush=True)
    print(macro.to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", nargs="+", choices=DOMAINS, default=DOMAINS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=REPORT_SEEDS, default=REPORT_SEEDS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.raw_dir.is_dir():
        raise FileNotFoundError(args.raw_dir)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "model_source_sha256": sha256(ROOT / "src/rul_tta/protocol_v1_models.py"),
                "variants": {
                    "dual_branch": "full two-branch architecture",
                    "temporal_only": "sensor feature vector zeroed before the original fusion head",
                    "sensor_only": "temporal feature vector zeroed before the original fusion head",
                    "single_receptive_field": "temporal branch replaced by one 128-channel causal k=9 convolution; sensor branch unchanged",
                },
                "evaluation": "official uncapped final-engine RUL, four target folds and five seeds",
                "training": {"optimizer": "Adam", "epochs": FROZEN_FIXED_EPOCHS, "batch_size": 64},
                "same_input": "source-fitted continuous condition residual, 30 cycles x 14 sensors",
                "dual_reference": "trained with the same frozen core as the other three variants",
                "experiment_script_sha256": sha256(Path(__file__)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for target in args.targets:
        base_config = ProtocolV1Config(
            target_domain=target, raw_dir=args.raw_dir.resolve(), output_dir=args.output,
            seed=REPORT_SEEDS[0], device=args.device, epochs=FROZEN_FIXED_EPOCHS,
        )
        fold = prepare_protocol_fold(base_config, preprocessing="continuous_comp")
        for seed in args.seeds:
            config = ProtocolV1Config(
                target_domain=target, raw_dir=args.raw_dir.resolve(), output_dir=args.output,
                seed=seed, device=args.device, epochs=FROZEN_FIXED_EPOCHS,
            )
            for variant in args.variants:
                run_dir = args.output / variant / target / f"seed_{seed}"
                audit_path = run_dir / "audit.json"
                if audit_path.is_file():
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                else:
                    audit = train_and_score(fold, config, variant, run_dir)
                print(
                    json.dumps(
                        {"target": target, "seed": seed, "variant": variant,
                         "rmse": audit["metrics"]["rmse"], "phm08": audit["metrics"]["phm08"],
                         "training_seconds": audit["training_seconds"]}
                    ),
                    flush=True,
                )
        del fold
        torch.cuda.empty_cache()
    aggregate(args.output)


if __name__ == "__main__":
    main()
