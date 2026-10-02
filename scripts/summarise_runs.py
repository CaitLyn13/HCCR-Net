"""Recompute paper metrics from saved final-engine predictions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from rul_tta.metrics import metric_dict
from rul_tta.protocol_v1_config import DOMAINS, REPORT_SEEDS


RUNS = {
    "HCCR-Net": ("main", "raw_compensated"),
    "ERM": ("main", "raw_no_condition"),
    "CORAL": ("baseline", "slow_coral"),
    "GroupDRO": ("baseline", "slow_group_dro"),
    "VREx": ("baseline", "slow_vrex"),
    "MS-DSN-style": ("baseline", "slow_ms_dsn"),
    "MDMT": ("mdmt", "mdmt"),
    "Operating variables as input": ("condition", "settings_input"),
    "Cluster mean": ("condition", "healthy_cluster_mean"),
    "Cluster mean + scale": ("condition", "healthy_cluster_zscore"),
    "RAFD-OCMM-style": ("condition", "rafd_ocmm_style"),
}
METRICS = ("rmse", "mae", "phm08", "signed_bias", "worst_10pct_engine_rmse")


def score(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if "is_final" in frame:
        frame = frame.loc[frame.is_final.astype(bool)]
    frame = frame.sort_values("engine_id")
    expected = np.arange(1, len(frame) + 1)
    if not np.array_equal(frame.engine_id.to_numpy(), expected):
        raise ValueError(f"Unexpected target-engine order in {path}")
    truth = frame.true_rul.to_numpy(np.float64)
    predicted = frame.prediction.to_numpy(np.float64)
    errors = predicted - truth
    tail = max(1, math.ceil(0.1 * len(errors)))
    return {
        **metric_dict(truth, predicted),
        "worst_10pct_engine_rmse": float(np.sqrt(np.mean(np.sort(errors**2)[-tail:]))),
    }


def summarise(run_root: Path, require_complete: bool) -> pd.DataFrame:
    rows = []
    for method, (family, variant) in RUNS.items():
        for target in DOMAINS:
            for seed in REPORT_SEEDS:
                prediction = run_root / family / variant / target / f"seed_{seed}" / "engine_predictions.csv"
                if not prediction.is_file():
                    if require_complete:
                        raise FileNotFoundError(prediction)
                    continue
                rows.append({"method": method, "target": target, "seed": seed, **score(prediction)})
    if not rows:
        raise FileNotFoundError(f"No final-engine predictions under {run_root}")
    result = pd.DataFrame(rows)
    if result.duplicated(["method", "target", "seed"]).any():
        raise ValueError("Duplicate method–target–seed result")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("results/paper_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    runs = summarise(args.run_root, args.require_complete)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "runs.csv", index=False)
    target_means = runs.groupby(["method", "target"], as_index=False)[list(METRICS)].mean()
    target_means.to_csv(args.output_dir / "target_means.csv", index=False)
    macro = target_means.groupby("method", as_index=False)[list(METRICS)].mean()
    macro.to_csv(args.output_dir / "macro_means.csv", index=False)
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
