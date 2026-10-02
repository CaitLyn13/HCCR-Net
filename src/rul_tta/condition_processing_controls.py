from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import (
    SourceData,
    TargetStream,
    _kmeans,
    add_train_rul,
    load_test_domain,
    load_train_domain,
    make_source_windows,
    make_target_stream,
    sensor_columns,
)
from .protocol_v1_config import DOMAINS, ProtocolV1Config
from .protocol_v1_data import (
    active_setting_dimensions,
    split_fit_calibration_engines,
)


SETTING_COLUMNS = ("setting1", "setting2", "setting3")
CONTROL_METHODS = (
    "settings_input",
    "healthy_cluster_mean",
    "healthy_cluster_zscore",
    "rafd_ocmm_style",
)
HEALTHY_RUL_THRESHOLD = 125.0
CLUSTER_COUNT = 6
MAPPING_SEED = 2026
OCMM_EPOCHS = 200


@dataclass(frozen=True)
class ConditionProcessingState:
    method: str
    sensor_columns: tuple[str, ...]
    model_columns: tuple[str, ...]
    setting_mean: np.ndarray
    setting_std: np.ndarray
    sensor_mean: np.ndarray
    sensor_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    cluster_centers: np.ndarray
    cluster_mean: np.ndarray
    cluster_std: np.ndarray
    ocmm_state: dict[str, np.ndarray]
    healthy_rows: int


@dataclass(frozen=True)
class ControlFoldData:
    fit: SourceData
    calibration: SourceData
    target: TargetStream
    state: ConditionProcessingState
    active_setting_dimensions: np.ndarray
    fit_engine_ids: tuple[int, ...]
    calibration_engine_ids: tuple[int, ...]


class RAFDOCMM(nn.Module):
    """RAFD Table-1 OCMM adapted from 4->14 to 3->14 dimensions."""

    def __init__(self, input_features: int = 3, output_features: int = 14):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_features),
        )

    def forward(self, settings: torch.Tensor) -> torch.Tensor:
        return self.network(settings)


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(std) & (std > 1e-8), std, 1.0).astype(np.float32)


def _settings_array(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, SETTING_COLUMNS].to_numpy(dtype=np.float32)


def _sensor_array(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, columns].to_numpy(dtype=np.float32)


def _normalise_settings(
    values: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return np.clip((values - mean) / std, -5.0, 5.0).astype(np.float32)


def _nearest_cluster(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.square(values[:, None, :] - centers[None, :, :]).sum(axis=2).argmin(axis=1)


def _healthy_frame(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    healthy = pd.concat(
        [frame.loc[frame["raw_rul"] >= HEALTHY_RUL_THRESHOLD] for frame in frames],
        ignore_index=True,
    )
    if healthy.empty:
        raise ValueError("No source fitting rows satisfy raw RUL >= 125.")
    return healthy


def fit_healthy_clusters(
    healthy_settings: np.ndarray,
    healthy_sensors: np.ndarray,
    clusters: int = CLUSTER_COUNT,
    seed: int = MAPPING_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = _kmeans(healthy_settings, clusters, seed)
    labels = _nearest_cluster(healthy_settings, centers)
    global_mean = healthy_sensors.mean(axis=0).astype(np.float32)
    global_std = _safe_std(healthy_sensors.std(axis=0, ddof=1))
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for cluster in range(clusters):
        selected = healthy_sensors[labels == cluster]
        if len(selected) == 0:
            means.append(global_mean)
            stds.append(global_std)
            continue
        means.append(selected.mean(axis=0).astype(np.float32))
        if len(selected) > 1:
            stds.append(_safe_std(selected.std(axis=0, ddof=1)))
        else:
            stds.append(global_std)
    return (
        centers.astype(np.float32),
        np.asarray(means, dtype=np.float32),
        np.asarray(stds, dtype=np.float32),
    )


def apply_cluster_transform(
    sensors: np.ndarray,
    settings: np.ndarray,
    centers: np.ndarray,
    cluster_mean: np.ndarray,
    cluster_std: np.ndarray,
    use_cluster_scale: bool,
) -> np.ndarray:
    labels = _nearest_cluster(settings, centers)
    residual = sensors - cluster_mean[labels]
    if use_cluster_scale:
        residual = residual / cluster_std[labels]
    return residual.astype(np.float32)


def fit_rafd_ocmm(
    settings: np.ndarray,
    sensors: np.ndarray,
    sensor_mean: np.ndarray,
    sensor_std: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    torch.manual_seed(MAPPING_SEED)
    np.random.seed(MAPPING_SEED)
    model = RAFDOCMM(settings.shape[1], sensors.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters())
    criterion = nn.L1Loss()
    x = torch.from_numpy(settings).to(device)
    y = torch.from_numpy((sensors - sensor_mean) / sensor_std).to(
        device, dtype=torch.float32
    )
    model.train()
    for _ in range(OCMM_EPOCHS):
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()
    return {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in model.state_dict().items()
    }


def apply_rafd_ocmm(
    settings: np.ndarray,
    state: dict[str, np.ndarray],
    sensor_mean: np.ndarray,
    sensor_std: np.ndarray,
    device: str,
) -> np.ndarray:
    model = RAFDOCMM(settings.shape[1], sensor_mean.shape[0])
    model.load_state_dict(
        {key: torch.from_numpy(value) for key, value in state.items()}
    )
    model.eval().to(device)
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(settings), 65536):
            batch = torch.from_numpy(settings[start : start + 65536]).to(device)
            output.append(model(batch).cpu().numpy())
    predicted_standard = np.concatenate(output, axis=0)
    return (predicted_standard * sensor_std + sensor_mean).astype(np.float32)


def fit_processing_state(
    frames: Iterable[pd.DataFrame],
    method: str,
    device: str,
) -> ConditionProcessingState:
    if method not in CONTROL_METHODS:
        raise ValueError(f"Unknown control method: {method}")
    frames = list(frames)
    sensors = sensor_columns()
    all_settings = np.concatenate([_settings_array(frame) for frame in frames])
    setting_mean = all_settings.mean(axis=0).astype(np.float32)
    setting_std = _safe_std(all_settings.std(axis=0, ddof=1))
    settings_z = _normalise_settings(all_settings, setting_mean, setting_std)
    all_sensors = np.concatenate([_sensor_array(frame, sensors) for frame in frames])
    sensor_mean = all_sensors.mean(axis=0).astype(np.float32)
    sensor_std = _safe_std(all_sensors.std(axis=0, ddof=1))
    healthy = _healthy_frame(frames)
    healthy_settings = _normalise_settings(
        _settings_array(healthy), setting_mean, setting_std
    )
    healthy_sensors = _sensor_array(healthy, sensors)
    centers = np.empty((0, 3), dtype=np.float32)
    cluster_mean = np.empty((0, len(sensors)), dtype=np.float32)
    cluster_std = np.empty((0, len(sensors)), dtype=np.float32)
    ocmm_state: dict[str, np.ndarray] = {}

    if method == "settings_input":
        transformed = (all_sensors - sensor_mean) / sensor_std
        model_columns = sensors + SETTING_COLUMNS
    elif method in {"healthy_cluster_mean", "healthy_cluster_zscore"}:
        centers, cluster_mean, cluster_std = fit_healthy_clusters(
            healthy_settings, healthy_sensors
        )
        transformed = apply_cluster_transform(
            all_sensors,
            settings_z,
            centers,
            cluster_mean,
            cluster_std,
            use_cluster_scale=method == "healthy_cluster_zscore",
        )
        model_columns = sensors
    else:
        ocmm_state = fit_rafd_ocmm(
            healthy_settings,
            healthy_sensors,
            sensor_mean,
            sensor_std,
            device,
        )
        response = apply_rafd_ocmm(
            settings_z, ocmm_state, sensor_mean, sensor_std, device
        )
        transformed = all_sensors - response
        model_columns = sensors

    output_mean = transformed.mean(axis=0).astype(np.float32)
    output_std = _safe_std(transformed.std(axis=0, ddof=1))
    return ConditionProcessingState(
        method=method,
        sensor_columns=sensors,
        model_columns=model_columns,
        setting_mean=setting_mean,
        setting_std=setting_std,
        sensor_mean=sensor_mean,
        sensor_std=sensor_std,
        output_mean=output_mean,
        output_std=output_std,
        cluster_centers=centers,
        cluster_mean=cluster_mean,
        cluster_std=cluster_std,
        ocmm_state=ocmm_state,
        healthy_rows=len(healthy),
    )


def transform_frame(
    frame: pd.DataFrame,
    state: ConditionProcessingState,
    device: str,
) -> pd.DataFrame:
    out = frame.copy()
    settings_z = _normalise_settings(
        _settings_array(out), state.setting_mean, state.setting_std
    )
    sensors = _sensor_array(out, state.sensor_columns)
    if state.method == "settings_input":
        transformed = (sensors - state.sensor_mean) / state.sensor_std
    elif state.method in {"healthy_cluster_mean", "healthy_cluster_zscore"}:
        transformed = apply_cluster_transform(
            sensors,
            settings_z,
            state.cluster_centers,
            state.cluster_mean,
            state.cluster_std,
            use_cluster_scale=state.method == "healthy_cluster_zscore",
        )
        transformed = (transformed - state.output_mean) / state.output_std
    elif state.method == "rafd_ocmm_style":
        response = apply_rafd_ocmm(
            settings_z,
            state.ocmm_state,
            state.sensor_mean,
            state.sensor_std,
            device,
        )
        transformed = (sensors - response - state.output_mean) / state.output_std
    else:
        raise ValueError(f"Unknown control method: {state.method}")
    transformed = np.clip(transformed, -5.0, 5.0).astype(np.float32)
    for index, column in enumerate(state.sensor_columns):
        out[column] = transformed[:, index]
    for index, column in enumerate(SETTING_COLUMNS):
        out[column] = settings_z[:, index]
    return out


def _combine(parts: list[tuple[np.ndarray, ...]]) -> SourceData:
    return SourceData(
        windows=np.concatenate([part[0] for part in parts]),
        rul=np.concatenate([part[1] for part in parts]),
        stage=np.zeros(sum(len(part[1]) for part in parts), dtype=np.int64),
        domain=np.concatenate([part[3] for part in parts]),
        engine=np.concatenate([part[4] for part in parts]),
        settings=np.concatenate([part[5] for part in parts]),
        cycle=np.concatenate([part[6] for part in parts]),
    )


def prepare_control_fold(
    config: ProtocolV1Config,
    method: str,
) -> ControlFoldData:
    config.validate()
    sources = tuple(domain for domain in DOMAINS if domain != config.target_domain)
    fit_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    fit_engine_ids: list[int] = []
    calibration_engine_ids: list[int] = []
    for domain_id, domain in enumerate(sources):
        frame = add_train_rul(load_train_domain(config.raw_dir, domain), config.rul_cap)
        fit_units, calibration_units = split_fit_calibration_engines(
            frame, config.calibration_fraction, config.split_seed + domain_id
        )
        fit_frame = frame.loc[frame["unit_id"].isin(fit_units)].copy()
        calibration_frame = frame.loc[frame["unit_id"].isin(calibration_units)].copy()
        fit_frames.append(fit_frame)
        calibration_frames.append(calibration_frame)
        fit_engine_ids.extend((domain_id * 10_000 + fit_units).tolist())
        calibration_engine_ids.extend((domain_id * 10_000 + calibration_units).tolist())

    state = fit_processing_state(fit_frames, method, config.device)
    fit_parts: list[tuple[np.ndarray, ...]] = []
    calibration_parts: list[tuple[np.ndarray, ...]] = []
    for domain_id, (fit_frame, calibration_frame) in enumerate(
        zip(fit_frames, calibration_frames)
    ):
        fit_transformed = transform_frame(fit_frame, state, config.device)
        calibration_transformed = transform_frame(
            calibration_frame, state, config.device
        )
        fit_parts.append(
            make_source_windows(
                fit_transformed, state.model_columns, config.window_size, domain_id
            )
        )
        calibration_parts.append(
            make_source_windows(
                calibration_transformed,
                state.model_columns,
                config.window_size,
                domain_id,
            )
        )

    target_raw = load_test_domain(config.raw_dir, config.target_domain)
    target_transformed = transform_frame(target_raw, state, config.device)
    target = make_target_stream(
        target_transformed, state.model_columns, config.window_size
    )
    return ControlFoldData(
        fit=_combine(fit_parts),
        calibration=_combine(calibration_parts),
        target=target,
        state=state,
        active_setting_dimensions=active_setting_dimensions(fit_frames),
        fit_engine_ids=tuple(sorted(fit_engine_ids)),
        calibration_engine_ids=tuple(sorted(calibration_engine_ids)),
    )


def save_processing_state(path: Path, state: ConditionProcessingState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "method": np.asarray(state.method),
        "sensor_columns": np.asarray(state.sensor_columns),
        "model_columns": np.asarray(state.model_columns),
        "setting_mean": state.setting_mean,
        "setting_std": state.setting_std,
        "sensor_mean": state.sensor_mean,
        "sensor_std": state.sensor_std,
        "output_mean": state.output_mean,
        "output_std": state.output_std,
        "cluster_centers": state.cluster_centers,
        "cluster_mean": state.cluster_mean,
        "cluster_std": state.cluster_std,
        "healthy_rows": np.asarray(state.healthy_rows),
        "ocmm_keys": np.asarray(tuple(state.ocmm_state)),
    }
    for index, key in enumerate(state.ocmm_state):
        payload[f"ocmm_value_{index}"] = state.ocmm_state[key]
    np.savez_compressed(path, **payload)
