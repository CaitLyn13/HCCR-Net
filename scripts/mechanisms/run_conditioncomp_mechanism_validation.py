from __future__ import annotations

"""ConditionComp mechanism validation.

Artifact contract: AUD-CCMV-001 / CODE-CCMV-001.
Inputs: source-only C-MAPSS folds and frozen raw_compensated checkpoints.
Outputs: reference-fit comparison, source probes, trajectory ordering, and
inference-only response-surface bootstrap stress results under this directory.
"""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rul_tta.data import (  # noqa: E402
    SourceData,
    SourceStatistics,
    TargetStream,
    _kmeans,
    add_train_rul,
    apply_condition_response,
    condition_basis,
    fit_source_statistics,
    load_target_rul,
    load_test_domain,
    load_train_domain,
    make_source_windows,
    make_target_stream,
    normalize,
    normalize_settings,
    sensor_columns,
    small_normal_equations,
    solve_small_linear_system,
)
from rul_tta.metrics import metric_dict  # noqa: E402
from rul_tta.protocol_v1_config import (  # noqa: E402
    DOMAINS,
    FROZEN_FIXED_EPOCHS,
    REPORT_SEEDS,
    ProtocolV1Config,
)
from rul_tta.protocol_v1_data import (  # noqa: E402
    ProtocolFoldData,
    _combine_source_parts,
    _subset_engines,
    active_setting_dimensions,
    fit_frozen_condition_statistics,
    split_fit_calibration_engines,
)
from rul_tta.protocol_v1_models import (  # noqa: E402
    ConditionCompRULNet,
    build_conditioncomp_rul_net,
)
from rul_tta.protocol_v1_train import train_final_p3  # noqa: E402


REFERENCES = ("healthy_only", "all_life", "matched_random")
REPRESENTATIONS = ("global_zscore",) + REFERENCES
SETTINGS = ("setting1", "setting2", "setting3")
MATCH_SEED = 1701
BOOTSTRAP_SEED = 2718
RIDGE_ALPHA = 1e-3
LATE_RUL = 40.0


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(value: object):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def source_frames(config: ProtocolV1Config):
    domains = tuple(domain for domain in DOMAINS if domain != config.target_domain)
    fit_frames, calibration_frames = [], []
    fit_ids, calibration_ids = [], []
    for domain_id, domain in enumerate(domains):
        frame = add_train_rul(load_train_domain(config.raw_dir, domain), config.rul_cap)
        fit_units, calibration_units = split_fit_calibration_engines(
            frame, config.calibration_fraction, config.split_seed + domain_id
        )
        fit_frames.append(_subset_engines(frame, fit_units))
        calibration_frames.append(_subset_engines(frame, calibration_units))
        fit_ids.extend((domain_id * 10_000 + fit_units).tolist())
        calibration_ids.extend((domain_id * 10_000 + calibration_units).tolist())
    return domains, fit_frames, calibration_frames, tuple(sorted(fit_ids)), tuple(sorted(calibration_ids))


def fit_coefficients(reference: pd.DataFrame, statistics: SourceStatistics, features: tuple[str, ...]) -> np.ndarray:
    settings = reference.loc[:, SETTINGS].to_numpy(dtype=np.float32)
    sensors = reference.loc[:, features].to_numpy(dtype=np.float32)
    design = condition_basis(settings, statistics.setting_mean, statistics.setting_std)
    regularizer = np.eye(10, dtype=np.float64) * 1e-3
    regularizer[0, 0] = 0.0
    gram, cross = small_normal_equations(design, sensors.astype(np.float64))
    return solve_small_linear_system(gram + regularizer, cross).astype(np.float32)


def select_matched_random(
    domains: tuple[str, ...],
    frames: list[pd.DataFrame],
    setting_mean: np.ndarray,
    setting_std: np.ndarray,
) -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(MATCH_SEED)
    selected_parts: list[pd.DataFrame] = []
    audit: list[dict] = []
    for domain, frame in zip(domains, frames):
        values = np.clip(
            (frame.loc[:, SETTINGS].to_numpy(dtype=np.float32) - setting_mean) / setting_std,
            -5.0,
            5.0,
        )
        clusters = 1 if domain in {"FD001", "FD003"} else 6
        if clusters == 1:
            labels = np.zeros(len(frame), dtype=np.int64)
        else:
            centers = _kmeans(values, clusters, MATCH_SEED)
            labels = np.square(values[:, None, :] - centers[None, :, :]).sum(axis=2).argmin(axis=1)
        healthy_mask = frame["raw_rul"].to_numpy(dtype=np.float32) >= 125.0
        for cluster in range(clusters):
            pool = np.flatnonzero(labels == cluster)
            count = int(np.sum(healthy_mask & (labels == cluster)))
            if count == 0:
                continue
            if count > len(pool):
                raise AssertionError("Matched-random count exceeds its all-life stratum.")
            chosen = rng.choice(pool, size=count, replace=False)
            selected_parts.append(frame.iloc[np.sort(chosen)].copy())
            audit.append(
                {
                    "domain": domain,
                    "cluster": cluster,
                    "all_life_rows": len(pool),
                    "healthy_rows": count,
                    "matched_random_rows": len(chosen),
                    "matched_random_healthy_fraction": float(np.mean(healthy_mask[chosen])),
                }
            )
    selected = pd.concat(selected_parts, ignore_index=True)
    expected = sum(int((frame["raw_rul"] >= 125.0).sum()) for frame in frames)
    if len(selected) != expected:
        raise AssertionError(f"Matched-random size {len(selected)} != healthy size {expected}.")
    return selected, audit


def fit_reference_statistics(
    domains: tuple[str, ...], frames: list[pd.DataFrame], reference: str
) -> tuple[SourceStatistics, dict]:
    features = sensor_columns()
    all_settings = pd.concat([frame.loc[:, SETTINGS] for frame in frames], ignore_index=True).to_numpy(np.float32)
    setting_mean = all_settings.mean(axis=0).astype(np.float32)
    setting_std = all_settings.std(axis=0, ddof=1).astype(np.float32)
    setting_std = np.where(setting_std > 1e-8, setting_std, 1.0).astype(np.float32)
    template = SourceStatistics(
        preprocessing="continuous_comp",
        mean=np.zeros(len(features), dtype=np.float32),
        std=np.ones(len(features), dtype=np.float32),
        feature_columns=features,
        setting_mean=setting_mean,
        setting_std=setting_std,
        condition_coefficients=np.zeros((10, len(features)), dtype=np.float32),
        regime_centers=np.empty((0, 3), dtype=np.float32),
        regime_mean=np.empty((0, len(features)), dtype=np.float32),
        regime_std=np.empty((0, len(features)), dtype=np.float32),
    )
    matched_audit: list[dict] = []
    if reference == "healthy_only":
        reference_frame = pd.concat([frame.loc[frame["raw_rul"] >= 125.0] for frame in frames], ignore_index=True)
    elif reference == "all_life":
        reference_frame = pd.concat(frames, ignore_index=True)
    elif reference == "matched_random":
        reference_frame, matched_audit = select_matched_random(domains, frames, setting_mean, setting_std)
    else:
        raise ValueError(reference)
    coefficients = fit_coefficients(reference_frame, template, features)
    all_sensors = pd.concat([frame.loc[:, features] for frame in frames], ignore_index=True).to_numpy(np.float32)
    residual = all_sensors - apply_condition_response(
        condition_basis(all_settings, setting_mean, setting_std), coefficients
    )
    residual_mean = residual.mean(axis=0).astype(np.float32)
    residual_std = residual.std(axis=0, ddof=1).astype(np.float32)
    residual_std = np.where(residual_std > 1e-8, residual_std, 1.0).astype(np.float32)
    stats = replace(template, mean=residual_mean, std=residual_std, condition_coefficients=coefficients)
    by_domain = []
    offset = 0
    for domain, frame in zip(domains, frames):
        if reference == "healthy_only":
            count = int((frame["raw_rul"] >= 125.0).sum())
        elif reference == "all_life":
            count = len(frame)
        else:
            count = sum(item["matched_random_rows"] for item in matched_audit if item["domain"] == domain)
        by_domain.append({"domain": domain, "all_life_rows": len(frame), "healthy_rows": int((frame["raw_rul"] >= 125.0).sum()), "reference_rows": count})
        offset += len(frame)
    audit = {
        "reference": reference,
        "reference_rows": len(reference_frame),
        "by_domain": by_domain,
        "matched_strata": matched_audit,
        "match_seed": MATCH_SEED if reference == "matched_random" else None,
    }
    if reference == "healthy_only":
        canonical = fit_frozen_condition_statistics(frames, features)
        max_diff = float(np.max(np.abs(canonical.condition_coefficients - coefficients)))
        audit["canonical_coefficient_max_abs_diff"] = max_diff
        if max_diff > 1e-6:
            raise AssertionError(f"Healthy-only coefficients do not reproduce canonical fit: {max_diff}")
    return stats, audit


def build_source_data(frames: list[pd.DataFrame], statistics: SourceStatistics, window_size: int) -> SourceData:
    parts = []
    for domain_id, frame in enumerate(frames):
        normalized = normalize_settings(normalize(frame, statistics), statistics)
        parts.append(make_source_windows(normalized, statistics.feature_columns, window_size, domain_id))
    return _combine_source_parts(parts)


def prepare_fold(config: ProtocolV1Config, representation: str, include_target: bool) -> tuple[ProtocolFoldData | None, dict]:
    domains, fit_frames, cal_frames, fit_ids, cal_ids = source_frames(config)
    if representation == "global_zscore":
        statistics = fit_source_statistics(fit_frames, sensor_columns(), preprocessing="global_zscore", seed=config.split_seed)
        audit = {"reference": representation, "reference_rows": sum(len(frame) for frame in fit_frames)}
    else:
        statistics, audit = fit_reference_statistics(domains, fit_frames, representation)
    fit = build_source_data(fit_frames, statistics, config.window_size)
    calibration = build_source_data(cal_frames, statistics, config.window_size)
    target = None
    if include_target:
        target_frame = normalize_settings(normalize(load_test_domain(config.raw_dir, config.target_domain), statistics), statistics)
        target = make_target_stream(target_frame, statistics.feature_columns, config.window_size)
    fold = ProtocolFoldData(
        fit=fit,
        calibration=calibration,
        target=target,  # type: ignore[arg-type]
        statistics=statistics,
        active_setting_dimensions=active_setting_dimensions(fit_frames),
        fit_engine_ids=fit_ids,
        calibration_engine_ids=cal_ids,
    )
    audit.update(
        {
            "target_domain": config.target_domain,
            "source_domains": domains,
            "fit_engines": len(fit_ids),
            "calibration_engines": len(cal_ids),
            "partitions_disjoint": not bool(set(fit_ids) & set(cal_ids)),
        }
    )
    return fold, audit


def window_descriptor(windows: np.ndarray) -> np.ndarray:
    mean = windows.mean(axis=1)
    last = windows[:, -1, :]
    time = np.arange(windows.shape[1], dtype=np.float32)
    time -= time.mean()
    denominator = float(np.sum(time * time))
    slope = np.einsum("nts,t->ns", windows, time, optimize=True) / denominator
    return np.concatenate([mean, last, slope], axis=1).astype(np.float64)


def domain_weights(domain: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(domain), dtype=np.float64)
    unique = np.unique(domain)
    for value in unique:
        mask = domain == value
        weights[mask] = 1.0 / max(1, int(mask.sum()))
    weights *= len(domain) / weights.sum()
    return weights


def fit_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float = RIDGE_ALPHA):
    x_mean = np.average(x, axis=0, weights=weights)
    x_var = np.average(np.square(x - x_mean), axis=0, weights=weights)
    x_std = np.sqrt(np.maximum(x_var, 1e-12))
    z = (x - x_mean) / x_std
    design = np.column_stack([np.ones(len(z)), z])
    y_2d = y[:, None] if y.ndim == 1 else y
    root_w = np.sqrt(weights)[:, None]
    gram, cross = small_normal_equations(design * root_w, y_2d * root_w)
    regularizer = np.eye(design.shape[1]) * alpha
    regularizer[0, 0] = 0.0
    coefficients = solve_small_linear_system(gram + regularizer, cross)
    return x_mean, x_std, coefficients, y.ndim == 1


def predict_ridge(model, x: np.ndarray) -> np.ndarray:
    x_mean, x_std, coefficients, squeeze = model
    design = np.column_stack([np.ones(len(x)), (x - x_mean) / x_std])
    prediction = np.sum(design[:, :, None] * coefficients[None, :, :], axis=1)
    return prediction[:, 0] if squeeze else prediction


def weighted_r2(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        y = y[:, None]
        prediction = prediction[:, None]
    mean = np.average(y, axis=0, weights=weights)
    total = np.sum(weights[:, None] * np.square(y - mean), axis=0)
    residual = np.sum(weights[:, None] * np.square(y - prediction), axis=0)
    return 1.0 - residual / np.maximum(total, 1e-12)


def ordered_kendall_tau(values: np.ndarray) -> float:
    """Kendall tau-a against strictly increasing cycle order."""
    concordant = 0
    discordant = 0
    for index in range(len(values) - 1):
        differences = values[index + 1 :] - values[index]
        concordant += int(np.sum(differences > 0.0))
        discordant += int(np.sum(differences < 0.0))
    pairs = len(values) * (len(values) - 1) // 2
    return float((concordant - discordant) / pairs) if pairs else 0.0


def run_probes_and_trajectory(raw_dir: Path, output_dir: Path) -> None:
    probe_rows, engine_rows, trajectory_rows, audit_rows = [], [], [], []
    for target in DOMAINS:
        print(f"[probes] target={target}", flush=True)
        config = ProtocolV1Config(target_domain=target, raw_dir=raw_dir, output_dir=output_dir, seed=REPORT_SEEDS[0], device="cpu")
        for representation in REPRESENTATIONS:
            print(f"[probes] representation={representation}", flush=True)
            fold, audit = prepare_fold(config, representation, include_target=False)
            print("[probes] fold ready", flush=True)
            assert fold is not None
            audit_rows.append(audit)
            x_fit = window_descriptor(fold.fit.windows)
            x_cal = window_descriptor(fold.calibration.windows)
            print("[probes] descriptors ready", flush=True)
            w_fit = domain_weights(fold.fit.domain)
            w_cal = domain_weights(fold.calibration.domain)
            condition_model = fit_ridge(x_fit, fold.fit.settings.astype(np.float64), w_fit)
            rul_model = fit_ridge(x_fit, fold.fit.rul.astype(np.float64), w_fit)
            print("[probes] ridge ready", flush=True)
            condition_prediction = predict_ridge(condition_model, x_cal)
            rul_prediction = predict_ridge(rul_model, x_cal).reshape(-1)
            condition_r2_dims = weighted_r2(fold.calibration.settings.astype(np.float64), condition_prediction, w_cal)
            rul_r2 = float(weighted_r2(fold.calibration.rul.astype(np.float64), rul_prediction, w_cal)[0])
            probe_rows.append(
                {
                    "target_fold": target,
                    "representation": representation,
                    "condition_r2": float(np.mean(condition_r2_dims)),
                    "condition_r2_setting1": condition_r2_dims[0],
                    "condition_r2_setting2": condition_r2_dims[1],
                    "condition_r2_setting3": condition_r2_dims[2],
                    "rul_r2": rul_r2,
                    "fit_windows": len(x_fit),
                    "calibration_windows": len(x_cal),
                    "descriptor_dim": x_fit.shape[1],
                }
            )
            # Label-free geometry after defining the source-only early-to-late direction.
            mean_x = np.average(x_fit, axis=0, weights=w_fit)
            std_x = np.sqrt(np.maximum(np.average(np.square(x_fit - mean_x), axis=0, weights=w_fit), 1e-12))
            z_fit = (x_fit - mean_x) / std_x
            z_cal = (x_cal - mean_x) / std_x
            early = fold.fit.rul >= 125.0
            late = fold.fit.rul <= LATE_RUL
            if not early.any() or not late.any():
                raise AssertionError("Trajectory direction requires nonempty early and late source windows.")
            early_center = z_fit[early].mean(axis=0)
            late_center = z_fit[late].mean(axis=0)
            direction = late_center - early_center
            norm = float(np.sqrt(np.sum(direction * direction)))
            if norm <= 1e-12:
                raise AssertionError("Degeneration direction collapsed.")
            direction /= norm
            score = np.sum((z_cal - early_center) * direction[None, :], axis=1)
            for engine in np.unique(fold.calibration.engine):
                mask = fold.calibration.engine == engine
                order = np.argsort(fold.calibration.cycle[mask])
                cycles = fold.calibration.cycle[mask][order]
                values = score[mask][order]
                rul = fold.calibration.rul[mask][order]
                tau = ordered_kendall_tau(values)
                stage_means = []
                for stage_mask in (rul >= 125.0, (rul < 125.0) & (rul > LATE_RUL), rul <= LATE_RUL):
                    stage_means.append(float(np.mean(values[stage_mask])) if stage_mask.any() else np.nan)
                ordered = bool(np.all(np.diff(stage_means) > 0)) if np.all(np.isfinite(stage_means)) else np.nan
                engine_rows.append(
                    {
                        "target_fold": target,
                        "representation": representation,
                        "engine_id": int(engine),
                        "windows": len(values),
                        "kendall_tau_cycle": tau,
                        "temporal_concordance": 0.5 * (tau + 1.0),
                        "early_score": stage_means[0],
                        "middle_score": stage_means[1],
                        "late_score": stage_means[2],
                        "three_stage_ordered": ordered,
                    }
                )
            subset = [row for row in engine_rows if row["target_fold"] == target and row["representation"] == representation]
            valid_order = [float(row["three_stage_ordered"]) for row in subset if not pd.isna(row["three_stage_ordered"])]
            trajectory_rows.append(
                {
                    "target_fold": target,
                    "representation": representation,
                    "engines": len(subset),
                    "kendall_tau_macro": float(np.mean([row["kendall_tau_cycle"] for row in subset])),
                    "temporal_concordance_macro": float(np.mean([row["temporal_concordance"] for row in subset])),
                    "three_stage_order_rate": float(np.mean(valid_order)) if valid_order else np.nan,
                    "engines_with_three_stages": len(valid_order),
                }
            )
    pd.DataFrame(probe_rows).to_csv(output_dir / "probe_fold_metrics.csv", index=False)
    pd.DataFrame(engine_rows).to_csv(output_dir / "trajectory_engine_metrics.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(output_dir / "trajectory_fold_summary.csv", index=False)
    save_json(output_dir / "source_mechanism_audit.json", audit_rows)
    probe = pd.DataFrame(probe_rows)
    trajectory = pd.DataFrame(trajectory_rows)
    probe.groupby("representation", sort=False).agg(
        condition_r2_mean=("condition_r2", "mean"),
        condition_r2_std=("condition_r2", "std"),
        rul_r2_mean=("rul_r2", "mean"),
        rul_r2_std=("rul_r2", "std"),
        folds=("target_fold", "count"),
    ).reset_index().to_csv(output_dir / "probe_macro_summary.csv", index=False)
    trajectory.groupby("representation", sort=False).agg(
        kendall_tau_mean=("kendall_tau_macro", "mean"),
        kendall_tau_std=("kendall_tau_macro", "std"),
        temporal_concordance_mean=("temporal_concordance_macro", "mean"),
        three_stage_order_rate_mean=("three_stage_order_rate", "mean"),
        folds=("target_fold", "count"),
    ).reset_index().to_csv(output_dir / "trajectory_macro_summary.csv", index=False)


@torch.no_grad()
def predict(model: ConditionCompRULNet, windows: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval().to(device)
    chunks = []
    for start in range(0, len(windows), batch_size):
        values = torch.from_numpy(windows[start : start + batch_size]).to(device)
        chunks.append(model(values)["prediction"].cpu().numpy())
    return np.concatenate(chunks)


def save_statistics(path: Path, statistics: SourceStatistics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        preprocessing=np.asarray(statistics.preprocessing), mean=statistics.mean, std=statistics.std,
        feature_columns=np.asarray(statistics.feature_columns), setting_mean=statistics.setting_mean,
        setting_std=statistics.setting_std, condition_coefficients=statistics.condition_coefficients,
    )


def score_target(run_dir: Path, raw_dir: Path, target: str, stream: TargetStream, predictions: np.ndarray, reference: str) -> dict:
    table = pd.DataFrame({
        "engine_id": stream.engine,
        "cycle": stream.cycle,
        "is_final": stream.is_final,
        "prediction": predictions,
    })
    table.to_csv(run_dir / "predictions_before_labels.csv", index=False)
    final = table.loc[table["is_final"]].sort_values("engine_id").reset_index(drop=True)
    truth = load_target_rul(raw_dir, target, expected_engines=len(final))
    if not np.array_equal(final["engine_id"].to_numpy(np.int64), np.arange(1, len(truth) + 1)):
        raise AssertionError("Target engine alignment failed.")
    final["true_rul"] = truth
    final["error"] = final["prediction"] - final["true_rul"]
    final["reference"] = reference
    final.to_csv(run_dir / "engine_predictions.csv", index=False)
    metrics = metric_dict(truth.astype(np.float64), final["prediction"].to_numpy(np.float64))
    save_json(run_dir / "metrics.json", metrics)
    return metrics


def run_reference_training(
    raw_dir: Path,
    output_dir: Path,
    device: str,
    force: bool,
    healthy_run_root: Path | None = None,
) -> None:
    rows = []
    current = healthy_run_root or (ROOT / "results" / "paper_runs" / "main" / "raw_compensated")
    for target in DOMAINS:
        base_config = ProtocolV1Config(target_domain=target, raw_dir=raw_dir, output_dir=output_dir, seed=REPORT_SEEDS[0], device=device, epochs=FROZEN_FIXED_EPOCHS)
        fold_cache: dict[str, ProtocolFoldData] = {}
        audit_cache: dict[str, dict] = {}
        for reference in REFERENCES:
            fold, audit = prepare_fold(base_config, reference, include_target=True)
            assert fold is not None
            fold_cache[reference] = fold
            audit_cache[reference] = audit
            save_json(output_dir / "reference_audits" / f"{target}_{reference}.json", audit)
        for seed in REPORT_SEEDS:
            frozen_metrics_path = current / target / f"seed_{seed}" / "metrics.json"
            frozen_preproc_path = current / target / f"seed_{seed}" / "preprocessing.npz"
            if not frozen_metrics_path.exists() or not frozen_preproc_path.exists():
                raise FileNotFoundError(frozen_metrics_path)
            frozen = json.loads(frozen_metrics_path.read_text(encoding="utf-8"))
            old = np.load(frozen_preproc_path)
            coefficient_diff = float(np.max(np.abs(old["condition_coefficients"] - fold_cache["healthy_only"].statistics.condition_coefficients)))
            if coefficient_diff > 1e-6:
                raise AssertionError(f"Frozen healthy preprocessing mismatch for {target}: {coefficient_diff}")
            rows.append({"target": target, "seed": seed, "reference": "healthy_only", **frozen, "source": str(frozen_metrics_path), "coefficient_max_abs_diff": coefficient_diff})
            for reference in ("all_life", "matched_random"):
                fold = fold_cache[reference]
                run_dir = output_dir / "reference_training" / reference / target / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                if metrics_path.exists() and not force:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    config = replace(base_config, seed=seed)
                    checkpoint = run_dir / "checkpoint.pt"
                    result = train_final_p3(fold.fit, config, FROZEN_FIXED_EPOCHS, checkpoint, representation_mode="raw_only")
                    save_statistics(run_dir / "preprocessing.npz", fold.statistics)
                    predictions = predict(result.model, fold.target.windows, config.tta_inference_batch_size, device)
                    metrics = score_target(run_dir, raw_dir, target, fold.target, predictions, reference)
                    save_json(run_dir / "audit.json", {
                        "target": target, "seed": seed, "reference": reference,
                        "fixed_epochs": FROZEN_FIXED_EPOCHS, "representation_mode": "raw_only",
                        "source_fit_engines": list(fold.fit_engine_ids),
                        "source_calibration_engines": list(fold.calibration_engine_ids),
                        "target_predictions_saved_before_labels": True,
                        "reference_audit": audit_cache[reference],
                        "config": {**asdict(config), "raw_dir": str(config.raw_dir), "output_dir": str(config.output_dir)},
                    })
                rows.append({"target": target, "seed": seed, "reference": reference, **metrics, "source": str(metrics_path), "coefficient_max_abs_diff": np.nan})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "reference_training_runs.csv", index=False)
    frame.groupby("reference", sort=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), mae_mean=("mae", "mean"),
        phm08_mean=("phm08", "mean"), runs=("rmse", "count")
    ).reset_index().to_csv(output_dir / "reference_training_macro.csv", index=False)
    frame.groupby(["reference", "target"], sort=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), runs=("rmse", "count")
    ).reset_index().to_csv(output_dir / "reference_training_by_target.csv", index=False)
    healthy = frame.loc[frame["reference"] == "healthy_only", ["target", "seed", "rmse"]].rename(columns={"rmse": "healthy_rmse"})
    paired_rows, paired_summary = [], []
    rng = np.random.default_rng(4401)
    for reference in ("all_life", "matched_random"):
        compared = frame.loc[frame["reference"] == reference, ["target", "seed", "rmse"]].rename(columns={"rmse": "comparison_rmse"})
        paired = healthy.merge(compared, on=["target", "seed"], validate="one_to_one")
        paired["comparison_reference"] = reference
        paired["healthy_advantage_rmse"] = paired["comparison_rmse"] - paired["healthy_rmse"]
        paired["healthy_wins"] = paired["healthy_advantage_rmse"] > 0.0
        paired_rows.append(paired)
        differences = paired["healthy_advantage_rmse"].to_numpy(np.float64)
        bootstrap = np.empty(10_000, dtype=np.float64)
        for index in range(len(bootstrap)):
            bootstrap[index] = np.mean(rng.choice(differences, size=len(differences), replace=True))
        paired_summary.append({
            "comparison_reference": reference,
            "pairs": len(paired),
            "healthy_wins": int(paired["healthy_wins"].sum()),
            "healthy_losses": int((~paired["healthy_wins"]).sum()),
            "mean_healthy_advantage_rmse": float(np.mean(differences)),
            "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_repetitions": len(bootstrap),
        })
    pd.concat(paired_rows, ignore_index=True).to_csv(output_dir / "reference_training_paired.csv", index=False)
    pd.DataFrame(paired_summary).to_csv(output_dir / "reference_training_paired_summary.csv", index=False)


def load_frozen_model(checkpoint: Path) -> ConditionCompRULNet:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_conditioncomp_rul_net(
        sensors=14,
        window_size=30,
        retained_bins=5,
        representation_mode="raw_only",
    )
    model.load_state_dict(payload["model_state"])
    return model


def bootstrap_coefficients(domains: tuple[str, ...], fit_frames: list[pd.DataFrame], central: SourceStatistics, repetitions: int) -> list[np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = [central.condition_coefficients.copy()]
    for _ in range(repetitions):
        parts = []
        for frame in fit_frames:
            engines = np.sort(frame["unit_id"].unique())
            sampled = rng.choice(engines, size=len(engines), replace=True)
            for new_index, engine in enumerate(sampled):
                part = frame.loc[frame["unit_id"] == engine].copy()
                part["unit_id"] = 1_000_000 + new_index
                parts.append(part.loc[part["raw_rul"] >= 125.0])
        reference = pd.concat(parts, ignore_index=True)
        output.append(fit_coefficients(reference, central, central.feature_columns))
    return output


def run_uncertainty_stress(
    raw_dir: Path,
    output_dir: Path,
    device: str,
    repetitions: int,
    checkpoint_root: Path | None = None,
    raw_baseline_root: Path | None = None,
) -> None:
    current_root = checkpoint_root or (ROOT / "results" / "paper_runs" / "main")
    baseline_root = raw_baseline_root or current_root
    all_scores = []
    for target in DOMAINS:
        config = ProtocolV1Config(target_domain=target, raw_dir=raw_dir, output_dir=output_dir, seed=REPORT_SEEDS[0], device=device)
        domains, fit_frames, _, _, _ = source_frames(config)
        central = fit_frozen_condition_statistics(fit_frames, sensor_columns())
        coefficients = bootstrap_coefficients(domains, fit_frames, central, repetitions)
        models = {
            seed: load_frozen_model(current_root / "raw_compensated" / target / f"seed_{seed}" / "checkpoints" / "p3_ablation.pt")
            for seed in REPORT_SEEDS
        }
        target_raw = load_test_domain(raw_dir, target)
        prediction_rows = []
        coefficient_rows = []
        for surface_id, coefficient in enumerate(coefficients):
            stats = replace(central, condition_coefficients=coefficient)
            normalized = normalize_settings(normalize(target_raw, stats), stats)
            stream = make_target_stream(normalized, stats.feature_columns, config.window_size)
            final_mask = stream.is_final
            final_windows = stream.windows[final_mask]
            final_engines = stream.engine[final_mask]
            for seed, model in models.items():
                values = predict(model, final_windows, config.tta_inference_batch_size, device)
                for engine, value in zip(final_engines, values):
                    prediction_rows.append({"target": target, "surface_id": surface_id, "seed": seed, "engine_id": int(engine), "prediction": float(value)})
            coefficient_rows.append({
                "target": target, "surface_id": surface_id,
                "is_center": surface_id == 0,
                "coefficient_l2_from_center": float(np.sqrt(np.sum(np.square(coefficient - central.condition_coefficients)))),
                "coefficient_max_abs_from_center": float(np.max(np.abs(coefficient - central.condition_coefficients))),
            })
        target_dir = output_dir / "response_uncertainty" / target
        target_dir.mkdir(parents=True, exist_ok=True)
        predictions = pd.DataFrame(prediction_rows)
        predictions.to_csv(target_dir / "predictions_before_labels.csv", index=False)
        pd.DataFrame(coefficient_rows).to_csv(target_dir / "surface_audit.csv", index=False)
        truth = load_target_rul(raw_dir, target, expected_engines=int(predictions["engine_id"].nunique()))
        truth_map = pd.Series(truth, index=np.arange(1, len(truth) + 1))
        predictions["true_rul"] = predictions["engine_id"].map(truth_map)
        if predictions["true_rul"].isna().any():
            raise AssertionError("Stress target alignment failed.")
        predictions.to_csv(target_dir / "scored_predictions.csv", index=False)
        center_lookup = predictions.loc[predictions["surface_id"] == 0].set_index(["seed", "engine_id"])["prediction"]
        for (surface_id, seed), part in predictions.groupby(["surface_id", "seed"], sort=True):
            y = part["true_rul"].to_numpy(np.float64)
            p = part["prediction"].to_numpy(np.float64)
            metrics = metric_dict(y, p)
            center_prediction = np.asarray([center_lookup.loc[(seed, engine)] for engine in part["engine_id"]], dtype=np.float64)
            frozen_file = current_root / "raw_compensated" / target / f"seed_{seed}" / "engine_predictions.csv"
            frozen = pd.read_csv(frozen_file).sort_values("engine_id")
            if surface_id == 0:
                max_diff = float(np.max(np.abs(p - frozen["prediction"].to_numpy(np.float64))))
                if max_diff > 1e-4:
                    raise AssertionError(f"Center stress view does not reproduce frozen prediction: {target}/{seed} {max_diff}")
            else:
                max_diff = np.nan
            baseline_metrics = json.loads((baseline_root / "raw_no_condition" / target / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
            all_scores.append({
                "target": target, "surface_id": surface_id, "seed": seed, **metrics,
                "mean_abs_prediction_change": float(np.mean(np.abs(p - center_prediction))),
                "max_abs_prediction_change": float(np.max(np.abs(p - center_prediction))),
                "center_reproduction_max_abs_diff": max_diff,
                "raw_no_condition_rmse": baseline_metrics["rmse"],
                "rmse_advantage_vs_raw_no_condition": baseline_metrics["rmse"] - metrics["rmse"],
            })
        save_json(target_dir / "audit.json", {
            "target": target, "source_domains": domains, "bootstrap_repetitions": repetitions,
            "bootstrap_seed": BOOTSTRAP_SEED, "engine_level_source_bootstrap": True,
            "central_residual_mean_std_frozen": True, "network_parameters_frozen": True,
            "predictions_saved_before_labels": True,
        })
    scores = pd.DataFrame(all_scores)
    center = scores.loc[scores["surface_id"] == 0, ["target", "seed", "rmse"]].rename(columns={"rmse": "center_rmse"})
    scores = scores.merge(center, on=["target", "seed"], how="left")
    scores["rmse_delta_vs_center"] = scores["rmse"] - scores["center_rmse"]
    scores.to_csv(output_dir / "response_uncertainty_runs.csv", index=False)
    stress = scores.loc[scores["surface_id"] > 0]
    summary = stress.groupby(["target", "seed"], sort=False).agg(
        rmse_delta_mean=("rmse_delta_vs_center", "mean"),
        rmse_delta_std=("rmse_delta_vs_center", "std"),
        rmse_delta_q025=("rmse_delta_vs_center", lambda x: float(np.quantile(x, 0.025))),
        rmse_delta_q975=("rmse_delta_vs_center", lambda x: float(np.quantile(x, 0.975))),
        rmse_delta_worst=("rmse_delta_vs_center", "max"),
        mean_abs_prediction_change=("mean_abs_prediction_change", "mean"),
        min_advantage_vs_raw=("rmse_advantage_vs_raw_no_condition", "min"),
        surfaces=("surface_id", "count"),
    ).reset_index()
    summary.to_csv(output_dir / "response_uncertainty_summary.csv", index=False)
    save_json(output_dir / "response_uncertainty_macro.json", {
        "runs": len(stress),
        "mean_rmse_delta": float(stress["rmse_delta_vs_center"].mean()),
        "q025_rmse_delta": float(stress["rmse_delta_vs_center"].quantile(0.025)),
        "q975_rmse_delta": float(stress["rmse_delta_vs_center"].quantile(0.975)),
        "worst_rmse_delta": float(stress["rmse_delta_vs_center"].max()),
        "minimum_advantage_vs_raw_no_condition": float(stress["rmse_advantage_vs_raw_no_condition"].min()),
        "all_stress_surfaces_better_than_raw_no_condition": bool((stress["rmse_advantage_vs_raw_no_condition"] > 0).all()),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("probes", "train", "stress", "all"), default="all")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "mechanism_validation",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-repetitions", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--healthy-run-root", type=Path)
    parser.add_argument("--raw-baseline-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"probes", "all"}:
        run_probes_and_trajectory(args.raw_dir.resolve(), args.output_dir.resolve())
    if args.stage in {"train", "all"}:
        run_reference_training(
            args.raw_dir.resolve(), args.output_dir.resolve(), args.device, args.force,
            args.healthy_run_root.resolve() if args.healthy_run_root else None,
        )
    if args.stage in {"stress", "all"}:
        run_uncertainty_stress(
            args.raw_dir.resolve(), args.output_dir.resolve(), args.device,
            args.bootstrap_repetitions,
            args.healthy_run_root.resolve().parent if args.healthy_run_root else None,
            args.raw_baseline_root.resolve() if args.raw_baseline_root else None,
        )
    completed_stages = [
        stage
        for stage, marker in (
            ("probes", "probe_fold_metrics.csv"),
            ("train", "reference_training_runs.csv"),
            ("stress", "response_uncertainty_runs.csv"),
        )
        if (args.output_dir / marker).exists()
    ]
    save_json(args.output_dir / "completion_audit.json", {
        "stage": "all" if len(completed_stages) == 3 else args.stage,
        "completed_stages": completed_stages,
        "references": REFERENCES,
        "representations": REPRESENTATIONS,
        "report_seeds": REPORT_SEEDS,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "artifact_contract": "AUD-CCMV-001",
    })


if __name__ == "__main__":
    main()
