"""MDMT 2026 baseline reimplementation under the frozen C-MAPSS protocol.

Artifact contract: CODE-MDMT-001 in
``docs/decisions/MDMT_UNIFIED_PROTOCOL_REPRODUCTION.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
from torch import nn

from .data import SourceData


PAPER_PARAMETER_COUNT = 68_739
REPRODUCED_PARAMETER_COUNT = 68_737
MDMT_STAGE_EPOCHS = (50, 40, 30)
MDMT_STAGE_LEARNING_RATES = (3e-3, 1.5e-3, 7.5e-4)
MDMT_BATCH_SIZE = 256
MDMT_TEMPERATURE = 0.05
MDMT_PERTURBATION = 0.05
MDMT_SOURCE_WEIGHT = 1.0
MDMT_KERNEL_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)


class MDMTBiLSTM(nn.Module):
    """Three-layer BiLSTM and MLP regressor matching MDMT's parameter scale."""

    def __init__(
        self,
        sensors: int = 14,
        hidden_per_direction: int = 32,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_per_direction * 2 != 64:
            raise ValueError("The registered MDMT reproduction fixes a 64-D BiLSTM output.")
        if layers != 3:
            raise ValueError("The registered MDMT reproduction fixes three BiLSTM layers.")
        self.feature_dim = hidden_per_direction * 2
        self.encoder = nn.LSTM(
            input_size=sensors,
            hidden_size=hidden_per_direction,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def encode(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("MDMT expects [batch, time, sensors] windows.")
        _, (hidden, _) = self.encoder(windows)
        return torch.cat((hidden[-2], hidden[-1]), dim=1)

    def predict_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.regressor(features).squeeze(-1)

    def forward(self, windows: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode(windows)
        return {
            "features": features,
            "prediction": self.predict_features(features),
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def mdmt_stage_schedule(source_domains: tuple[int, ...]) -> tuple[dict[str, int | float], ...]:
    """Three-source schedule: initial pair, new-domain replay, rotated anchor."""
    if len(source_domains) != 3 or len(set(source_domains)) != 3:
        raise ValueError("The unified C-MAPSS MDMT protocol requires exactly three sources.")
    d0, d1, d2 = sorted(int(value) for value in source_domains)
    pairs = ((d0, d1), (d0, d2), (d1, d2))
    return tuple(
        {
            "stage": index + 1,
            "anchor_domain": anchor,
            "new_domain": new,
            "epochs": MDMT_STAGE_EPOCHS[index],
            "learning_rate": MDMT_STAGE_LEARNING_RATES[index],
        }
        for index, (anchor, new) in enumerate(pairs)
    )


def _squared_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cdist(left, right, p=2).square()


def estimate_mmd_bandwidth(*features: torch.Tensor) -> torch.Tensor:
    combined = torch.cat(features, dim=0)
    if len(combined) < 2:
        return combined.new_tensor(1.0)
    distances = torch.pdist(combined, p=2).square()
    positive = distances[distances > 1e-12]
    if not len(positive):
        return combined.new_tensor(1.0)
    return positive.median().clamp_min(1e-6)


def multi_kernel_mmd(
    left: torch.Tensor,
    right: torch.Tensor,
    bandwidth: torch.Tensor | None = None,
    multipliers: tuple[float, ...] = MDMT_KERNEL_MULTIPLIERS,
) -> torch.Tensor:
    """Biased MK-MMD squared; stable and exactly zero for identical tensors."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("MK-MMD expects two 2-D feature matrices with equal width.")
    if bandwidth is None:
        bandwidth = estimate_mmd_bandwidth(left, right)

    def kernel(distance: torch.Tensor) -> torch.Tensor:
        terms = [
            torch.exp(-distance / (2.0 * bandwidth * multiplier))
            for multiplier in multipliers
        ]
        return torch.stack(terms, dim=0).mean(dim=0)

    value = (
        kernel(_squared_distance(left, left)).mean()
        + kernel(_squared_distance(right, right)).mean()
        - 2.0 * kernel(_squared_distance(left, right)).mean()
    )
    return value.clamp_min(0.0)


@dataclass(frozen=True)
class MixRatioUpdate:
    value: float
    center: float
    q: float


def update_mix_ratio(
    previous: float,
    distance_anchor: float,
    distance_new: float,
    step: int,
    total_steps: int,
    rng: np.random.Generator,
    temperature: float = MDMT_TEMPERATURE,
    perturbation: float = MDMT_PERTURBATION,
) -> MixRatioUpdate:
    if total_steps <= 0 or not 1 <= step <= total_steps:
        raise ValueError("step must be in [1, total_steps].")
    denominator = distance_anchor + distance_new * temperature + 1e-12
    q = float(np.exp(-distance_anchor / denominator))
    center = float((step / total_steps) * (1.0 - q) + q * previous)
    sampled = float(rng.uniform(center - perturbation, center + perturbation))
    return MixRatioUpdate(
        value=float(np.clip(sampled, 0.0, 1.0)),
        center=float(np.clip(center, 0.0, 1.0)),
        q=q,
    )


def _draw_cyclic(
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    state: dict[str, np.ndarray | int],
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    remaining = batch_size
    while remaining:
        order = state["order"]
        position = int(state["position"])
        assert isinstance(order, np.ndarray)
        available = len(order) - position
        take = min(remaining, available)
        pieces.append(order[position : position + take])
        position += take
        remaining -= take
        if position == len(order):
            order = rng.permutation(indices)
            position = 0
        state["order"] = order
        state["position"] = position
    return np.concatenate(pieces)


def _epoch_pair_batches(
    anchor_indices: np.ndarray,
    new_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    steps = int(ceil(max(len(anchor_indices), len(new_indices)) / batch_size))
    anchor_state: dict[str, np.ndarray | int] = {
        "order": rng.permutation(anchor_indices),
        "position": 0,
    }
    new_state: dict[str, np.ndarray | int] = {
        "order": rng.permutation(new_indices),
        "position": 0,
    }
    for _ in range(steps):
        yield (
            _draw_cyclic(anchor_indices, batch_size, rng, anchor_state),
            _draw_cyclic(new_indices, batch_size, rng, new_state),
        )


def _tensor_batch(
    source: SourceData,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    windows = torch.from_numpy(source.windows[indices]).to(
        device, non_blocking=non_blocking
    )
    target = torch.from_numpy(source.rul[indices]).to(
        device, non_blocking=non_blocking
    )
    return windows, target


def train_mdmt(
    source: SourceData,
    seed: int,
    device: str,
) -> tuple[MDMTBiLSTM, list[dict[str, float | int]]]:
    """Train the frozen three-stage MDMT schedule on source fitting windows."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)
    torch_device = torch.device(device)
    model = MDMTBiLSTM(sensors=source.windows.shape[2]).to(torch_device)
    if trainable_parameter_count(model) != REPRODUCED_PARAMETER_COUNT:
        raise RuntimeError("MDMT parameter-count audit failed before training.")

    schedule = mdmt_stage_schedule(tuple(int(v) for v in np.unique(source.domain)))
    optimizer = torch.optim.Adam(model.parameters(), lr=MDMT_STAGE_LEARNING_RATES[0])
    mix_ratio = float(rng.beta(2.0, 2.0))
    history: list[dict[str, float | int]] = []

    for stage in schedule:
        anchor_domain = int(stage["anchor_domain"])
        new_domain = int(stage["new_domain"])
        epochs = int(stage["epochs"])
        learning_rate = float(stage["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        anchor_indices = np.flatnonzero(source.domain == anchor_domain)
        new_indices = np.flatnonzero(source.domain == new_domain)
        if not len(anchor_indices) or not len(new_indices):
            raise RuntimeError("A registered MDMT source pair is empty.")
        steps_per_epoch = int(
            ceil(max(len(anchor_indices), len(new_indices)) / MDMT_BATCH_SIZE)
        )
        total_steps = epochs * steps_per_epoch
        stage_step = 0

        for epoch in range(1, epochs + 1):
            model.train()
            totals = {
                "loss": 0.0,
                "mix_input": 0.0,
                "mix_feature": 0.0,
                "source": 0.0,
                "lambda": 0.0,
                "q": 0.0,
                "mmd_anchor": 0.0,
                "mmd_new": 0.0,
            }
            batches = 0
            for anchor_batch, new_batch in _epoch_pair_batches(
                anchor_indices,
                new_indices,
                MDMT_BATCH_SIZE,
                rng,
            ):
                stage_step += 1
                anchor_x, anchor_y = _tensor_batch(source, anchor_batch, torch_device)
                new_x, new_y = _tensor_batch(source, new_batch, torch_device)
                anchor_features = model.encode(anchor_x)
                new_features = model.encode(new_x)

                with torch.no_grad():
                    provisional = (
                        mix_ratio * anchor_features.detach()
                        + (1.0 - mix_ratio) * new_features.detach()
                    )
                    bandwidth = estimate_mmd_bandwidth(
                        anchor_features.detach(), new_features.detach(), provisional
                    )
                    distance_anchor = multi_kernel_mmd(
                        anchor_features.detach(), provisional, bandwidth
                    )
                    distance_new = multi_kernel_mmd(
                        new_features.detach(), provisional, bandwidth
                    )
                    update = update_mix_ratio(
                        mix_ratio,
                        float(distance_anchor),
                        float(distance_new),
                        stage_step,
                        total_steps,
                        rng,
                    )
                    mix_ratio = update.value

                mixed_target = mix_ratio * anchor_y + (1.0 - mix_ratio) * new_y
                mixed_input = mix_ratio * anchor_x + (1.0 - mix_ratio) * new_x
                mixed_features = (
                    mix_ratio * anchor_features + (1.0 - mix_ratio) * new_features
                )
                anchor_prediction = model.predict_features(anchor_features)
                new_prediction = model.predict_features(new_features)
                input_prediction = model(mixed_input)["prediction"]
                feature_prediction = model.predict_features(mixed_features)

                source_loss = (
                    (anchor_prediction - anchor_y).square().mean()
                    + (new_prediction - new_y).square().mean()
                )
                mix_input_loss = (input_prediction - mixed_target).square().mean()
                mix_feature_loss = (feature_prediction - mixed_target).square().mean()
                loss = (
                    mix_input_loss
                    + mix_feature_loss
                    + MDMT_SOURCE_WEIGHT * source_loss
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("MDMT produced a non-finite training loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("MDMT produced a non-finite gradient.")
                optimizer.step()

                totals["loss"] += float(loss.detach())
                totals["mix_input"] += float(mix_input_loss.detach())
                totals["mix_feature"] += float(mix_feature_loss.detach())
                totals["source"] += float(source_loss.detach())
                totals["lambda"] += mix_ratio
                totals["q"] += update.q
                totals["mmd_anchor"] += float(distance_anchor)
                totals["mmd_new"] += float(distance_new)
                batches += 1

            row: dict[str, float | int] = {
                "stage": int(stage["stage"]),
                "anchor_domain": anchor_domain,
                "new_domain": new_domain,
                "epoch": epoch,
                "epochs_in_stage": epochs,
                "learning_rate": learning_rate,
                "batches": batches,
            }
            row.update({key: value / batches for key, value in totals.items()})
            history.append(row)
    return model, history
