"""Source-calibration audit of RUL-by-condition interactions in sensor inputs.

This source-only diagnostic reads C-MAPSS training trajectories and refits
the same healthy-reference preprocessing used by the prediction network.
Each regression is fitted on source-fitting engines and scored on disjoint
source-calibration engines within FD002 or FD004. The target test files and
official target RUL labels are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

from rul_tta.data import (  # noqa: E402
    NORMALIZATION_CLIP,
    _kmeans,
    add_train_rul,
    apply_condition_response,
    condition_basis,
    load_train_domain,
    sensor_columns,
    small_normal_equations,
    solve_small_linear_system,
)
from rul_tta.protocol_v1_config import (  # noqa: E402
    DOMAINS,
    SPLIT_SEED,
)
from rul_tta.protocol_v1_data import fit_frozen_condition_statistics, split_fit_calibration_engines


SETTINGS = ("setting1", "setting2", "setting3")
MULTI_CONDITION_SOURCES = ("FD002", "FD004")
REPRESENTATIONS = ("observed", "unclipped_residual", "network_input")


def source_partitions(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_ids, calibration_ids = split_fit_calibration_engines(frame, 0.15, seed)
    fit = frame.loc[frame.unit_id.isin(fit_ids)].copy()
    calibration = frame.loc[frame.unit_id.isin(calibration_ids)].copy()
    assert set(fit.unit_id.unique()).isdisjoint(calibration.unit_id.unique())
    return fit, calibration


def designs(
    frame: pd.DataFrame,
    setting_mean: np.ndarray,
    setting_std: np.ndarray,
    cluster_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Additive health/condition basis and its prespecified interactions."""
    raw_rul = frame.raw_rul.to_numpy(np.float64)
    if np.any((raw_rul < 0) | (raw_rul >= 125)):
        raise ValueError("The interaction probe uses uncapped raw RUL in [0, 125).")
    h = (125.0 - raw_rul) / 125.0
    u = frame.loc[:, SETTINGS].to_numpy(np.float64)
    basis = condition_basis(u, setting_mean, setting_std)
    u_std = basis[:, 1:4]
    additive = np.column_stack((h, h**2, h**3, basis[:, 1:]))
    if cluster_centers is not None:
        labels = np.square(u_std[:, None, :] - cluster_centers[None, :, :]).sum(axis=2).argmin(axis=1)
        additive = np.column_stack((additive, np.eye(len(cluster_centers))[labels, 1:]))
    interactions = np.column_stack((h[:, None] * u_std, (h**2)[:, None] * u_std))
    return additive, np.column_stack((additive, interactions))


def representations(frame: pd.DataFrame, statistics: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    observed = frame.loc[:, sensor_columns()].to_numpy(np.float64)
    basis = condition_basis(
        frame.loc[:, SETTINGS].to_numpy(np.float64),
        statistics["setting_mean"],
        statistics["setting_std"],
    )
    residual = observed - apply_condition_response(basis, statistics["condition_coefficients"])
    network_input = np.clip(
        (residual - statistics["mean"]) / statistics["std"],
        -NORMALIZATION_CLIP,
        NORMALIZATION_CLIP,
    )
    return {
        "observed": observed,
        "unclipped_residual": residual,
        "network_input": network_input,
    }


def demean_by_engine(values: np.ndarray, engines: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for engine in np.unique(engines):
        selected = engines == engine
        result[selected] = values[selected] - values[selected].mean(axis=0)
    return result


def engine_equal_weights(engines: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(engines, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def small_matrix_rank(matrix: np.ndarray) -> int:
    """Rank of a small symmetric Gram matrix without Windows LAPACK."""
    work = np.asarray(matrix, dtype=np.float64).copy()
    tolerance = max(1e-10, float(np.max(np.abs(work))) * 1e-9)
    rank = 0
    for column in range(work.shape[1]):
        pivot = rank + int(np.argmax(np.abs(work[rank:, column])))
        if abs(work[pivot, column]) <= tolerance:
            continue
        work[[rank, pivot]] = work[[pivot, rank]]
        work[rank] /= work[rank, column]
        for row in range(rank + 1, work.shape[0]):
            work[row] -= work[row, column] * work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def fit_and_score(
    fit_add: np.ndarray,
    fit_full: np.ndarray,
    fit_y: np.ndarray,
    fit_engine: np.ndarray,
    cal_add: np.ndarray,
    cal_full: np.ndarray,
    cal_y: np.ndarray,
    cal_engine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return per-engine, per-sensor held-out MSE for additive/full probes."""
    fit_designs = (demean_by_engine(fit_add, fit_engine), demean_by_engine(fit_full, fit_engine))
    cal_designs = (demean_by_engine(cal_add, cal_engine), demean_by_engine(cal_full, cal_engine))
    y_fit = demean_by_engine(fit_y, fit_engine)
    y_cal = demean_by_engine(cal_y, cal_engine)
    weights = np.sqrt(engine_equal_weights(fit_engine))[:, None]
    mse = []
    ranks = []
    for x_fit, x_cal in zip(fit_designs, cal_designs):
        scale = np.sqrt(np.average(x_fit**2, axis=0, weights=weights[:, 0] ** 2))
        scale = np.where(scale > 1e-12, scale, 1.0)
        x_fit_scaled = x_fit / scale
        x_cal_scaled = x_cal / scale
        gram, cross = small_normal_equations(x_fit_scaled * weights, y_fit * weights)
        rank = small_matrix_rank(gram)
        ridge = np.eye(len(gram), dtype=np.float64) * (1e-8 * max(1.0, float(np.trace(gram))))
        coefficients = solve_small_linear_system(gram + ridge, cross)
        error_sq = (y_cal - apply_condition_response(x_cal_scaled, coefficients)) ** 2
        per_engine = np.stack(
            [error_sq[cal_engine == engine].mean(axis=0) for engine in np.unique(cal_engine)]
        )
        mse.append(per_engine)
        ranks.append(int(rank))
    return mse[0], mse[1], ranks[0], ranks[1]


def summarise_gain(
    mse_add: np.ndarray, mse_full: np.ndarray, *, bootstrap_seed: int, replicates: int
) -> dict[str, object]:
    def gain(indices: np.ndarray) -> np.ndarray:
        a = mse_add[indices].mean(axis=0)
        b = mse_full[indices].mean(axis=0)
        return 1.0 - b / np.maximum(a, 1e-12)

    observed = gain(np.arange(len(mse_add)))
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(mse_add), size=len(mse_add))
        draws[index] = float(np.mean(gain(sampled)))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_sensor_relative_mse_gain": float(observed.mean()),
        "median_sensor_relative_mse_gain": float(np.median(observed)),
        "positive_sensors": int(np.sum(observed > 0.0)),
        "sensor_gains": observed.tolist(),
        "engine_bootstrap_95_low": float(low),
        "engine_bootstrap_95_high": float(high),
    }


def run(raw_dir: Path, output_dir: Path, replicates: int) -> None:
    if replicates < 100:
        raise ValueError("At least 100 engine-bootstrap draws are required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    engine_rows: list[dict[str, object]] = []
    fold_audit: list[dict[str, object]] = []
    cached_domains: dict[str, pd.DataFrame] = {}
    for target in DOMAINS:
        source_domains = tuple(domain for domain in DOMAINS if domain != target)
        partitions = {}
        fit_frames = []
        for source_index, source_domain in enumerate(source_domains):
            if source_domain not in cached_domains:
                cached_domains[source_domain] = add_train_rul(
                    load_train_domain(raw_dir, source_domain), 125.0
                )
            full = cached_domains[source_domain]
            fit, calibration = source_partitions(full, SPLIT_SEED + source_index)
            partitions[source_domain] = (fit, calibration)
            fit_frames.append(fit)
        frozen = fit_frozen_condition_statistics(fit_frames, sensor_columns())
        statistics = {
            name: getattr(frozen, name)
            for name in ("mean", "std", "setting_mean", "setting_std", "condition_coefficients")
        }
        preprocessing_sha256 = hashlib.sha256(
            np.ascontiguousarray(frozen.condition_coefficients).tobytes()
        ).hexdigest()
        for source_index, source_domain in enumerate(source_domains):
            fit, calibration = partitions[source_domain]
            if source_domain not in MULTI_CONDITION_SOURCES:
                continue
            full_fit_settings = fit.loc[:, SETTINGS].to_numpy(np.float64)
            normalized_fit_settings = np.clip(
                (full_fit_settings - statistics["setting_mean"]) / statistics["setting_std"],
                -NORMALIZATION_CLIP, NORMALIZATION_CLIP,
            )
            cluster_centers = _kmeans(normalized_fit_settings, 6, 1701).astype(np.float64)
            fit = fit.loc[fit.raw_rul < 125.0].copy()
            calibration = calibration.loc[calibration.raw_rul < 125.0].copy()
            fit_engine = fit.unit_id.to_numpy(np.int64)
            cal_engine = calibration.unit_id.to_numpy(np.int64)
            fit_repr = representations(fit, statistics)
            cal_repr = representations(calibration, statistics)
            stratum: dict[str, object] = {
                "target_fold": target,
                "source_subset": source_domain,
                "fit_engines": int(len(np.unique(fit_engine))),
                "calibration_engines": int(len(np.unique(cal_engine))),
                "fit_rows": int(len(fit)),
                "calibration_rows": int(len(calibration)),
                "response_coefficients_sha256": preprocessing_sha256,
            }
            for condition_control, centers in (("quadratic", None), ("quadratic_plus_clusters", cluster_centers)):
                fit_add, fit_full = designs(fit, statistics["setting_mean"], statistics["setting_std"], centers)
                cal_add, cal_full = designs(calibration, statistics["setting_mean"], statistics["setting_std"], centers)
                gains: dict[str, float] = {}
                for representation in REPRESENTATIONS:
                    mse_add, mse_full, rank_add, rank_full = fit_and_score(
                        fit_add, fit_full, fit_repr[representation], fit_engine,
                        cal_add, cal_full, cal_repr[representation], cal_engine,
                    )
                    score = summarise_gain(
                        mse_add,
                        mse_full,
                        bootstrap_seed=2026 + DOMAINS.index(target) * 10 + source_index,
                        replicates=replicates,
                    )
                    gains[representation] = float(score["mean_sensor_relative_mse_gain"])
                    summaries.append({
                        **{key: stratum[key] for key in (
                            "target_fold", "source_subset", "fit_engines", "calibration_engines",
                            "fit_rows", "calibration_rows"
                        )},
                        "condition_control": condition_control,
                        "additive_terms": int(fit_add.shape[1]),
                        "interaction_terms": int(fit_full.shape[1] - fit_add.shape[1]),
                        "representation": representation,
                        "rank_additive": rank_add,
                        "rank_with_interactions": rank_full,
                        **{key: value for key, value in score.items() if key != "sensor_gains"},
                    })
                    for engine_idx, engine in enumerate(np.unique(cal_engine)):
                        for sensor_idx, sensor in enumerate(sensor_columns()):
                            engine_rows.append({
                                "target_fold": target,
                                "source_subset": source_domain,
                                "condition_control": condition_control,
                                "representation": representation,
                                "engine_id": int(engine),
                                "sensor": sensor,
                                "mse_additive": float(mse_add[engine_idx, sensor_idx]),
                                "mse_interaction": float(mse_full[engine_idx, sensor_idx]),
                            })
                    print(
                        f"{target}/{source_domain}/{condition_control}/{representation}: "
                        f"gain={score['mean_sensor_relative_mse_gain']:.5f}, "
                        f"CI=[{score['engine_bootstrap_95_low']:.5f},"
                        f"{score['engine_bootstrap_95_high']:.5f}], "
                        f"positive_sensors={score['positive_sensors']}/14",
                        flush=True,
                    )
                stratum[f"raw_minus_unclipped_gain_{condition_control}"] = (
                    gains["observed"] - gains["unclipped_residual"]
                )
            fold_audit.append(stratum)
    pd.DataFrame(summaries).to_csv(output_dir / "probe_summary.csv", index=False)
    pd.DataFrame(engine_rows).to_csv(output_dir / "calibration_engine_mse.csv", index=False)
    (output_dir / "audit.json").write_text(
        json.dumps({
            "scope": "source-fit/source-calibration only; no target test data or labels",
            "reference_version": "source-only frozen conditioner fitted from the source-engine split",
            "response_reference": "source-fitted healthy-reference preprocessing",
            "target_folds": list(DOMAINS),
            "interaction_sources": list(MULTI_CONDITION_SOURCES),
            "health_proxy": "uncapped source raw RUL below 125 cycles",
            "additive": "engine-centred cubic health proxy + quadratic operating-condition basis; stricter check adds six source-fit condition-cluster fixed effects",
            "increment": "health proxy x three standardised settings, linear and quadratic",
            "score": "held-out engine-equal relative MSE gain, sensor macro-mean",
            "uncertainty": f"paired engine bootstrap, {replicates} draws, within each stratum",
            "interpretation_limit": "statistical interaction; not identified physical degradation or proof of network use",
            "strata": fold_audit,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/source_condition_interaction_probe",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    run(args.raw_dir.resolve(), args.output_dir.resolve(), args.bootstrap_replicates)


if __name__ == "__main__":
    main()
