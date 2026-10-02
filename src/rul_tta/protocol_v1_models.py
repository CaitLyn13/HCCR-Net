from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FixedLowPassProjector(nn.Module):
    """Fixed rFFT low-pass used only by the registered 2 x 2 ablation."""

    def __init__(self, window_size: int = 30, retained_bins: int = 5):
        super().__init__()
        if window_size != 30:
            raise ValueError("Protocol v1.0 fixes window_size=30.")
        if retained_bins not in {3, 5, 7}:
            raise ValueError("retained_bins must be one of 3, 5, 7.")
        self.window_size = int(window_size)
        self.retained_bins = int(retained_bins)

    def forward(self, values: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(values, dim=1)
        if spectrum.shape[1] != 16:
            raise RuntimeError(f"Expected 16 rFFT bins, got {spectrum.shape[1]}.")
        retained = torch.zeros_like(spectrum)
        retained[:, : self.retained_bins] = spectrum[:, : self.retained_bins]
        return torch.fft.irfft(retained, n=self.window_size, dim=1)


class CausalConvSummary(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, values: Tensor) -> Tensor:
        sequence = values.transpose(1, 2)
        sequence = F.pad(sequence, (self.kernel_size - 1, 0))
        sequence = self.conv(sequence).transpose(1, 2)
        sequence = F.gelu(self.norm(sequence))
        return sequence[:, -1]


class MultiReceptiveFieldCausalEncoder(nn.Module):
    """Four parallel causal 1-D convolutions with receptive fields 3/5/9/15."""

    def __init__(self, sensors: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [CausalConvSummary(sensors, 32, kernel) for kernel in (3, 5, 9, 15)]
        )

    def forward(self, residual: Tensor) -> Tensor:
        return torch.cat([branch(residual) for branch in self.branches], dim=1)


class SensorSelfAttentionEncoder(nn.Module):
    """Represent each sensor trajectory as a token and aggregate sensor relations."""

    def __init__(self, window_size: int, embedding: int = 128, heads: int = 4):
        super().__init__()
        self.projection = nn.Linear(window_size, embedding)
        self.norm = nn.LayerNorm(embedding)
        self.attention = nn.MultiheadAttention(embedding, heads, batch_first=True)

    def forward(self, residual: Tensor) -> Tensor:
        sensor_tokens = self.norm(self.projection(residual.transpose(1, 2)))
        attended, _ = self.attention(
            sensor_tokens, sensor_tokens, sensor_tokens, need_weights=False
        )
        return attended.mean(dim=1)


def _protocol_v1_fusion_initialization(sensors: int) -> tuple[Tensor, Tensor]:
    """Reproduce the frozen protocol-v1 parameter-initialisation stream.

    These temporary draws preserve exact seeded retraining compatibility with
    the registered experiments. They are not parameters, branches or forward
    computations of :class:`ConditionCompRULNet`.
    """

    for shape, fan_in in (
        ((64, sensors, 3), sensors * 3),
        ((64,), sensors * 3),
        ((64, 64, 3), 64 * 3),
        ((64,), 64 * 3),
    ):
        placeholder = torch.empty(shape)
        if len(shape) == 3:
            nn.init.kaiming_uniform_(placeholder, a=math.sqrt(5))
        else:
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(placeholder, -bound, bound)
    legacy_fusion = nn.Linear(128 + 128 + 64, 128)
    return (
        legacy_fusion.weight[:, : 128 + 128].detach().clone(),
        legacy_fusion.bias.detach().clone(),
    )


class ConditionCompRULNet(nn.Module):
    """RUL regressor for source-fitted condition residuals.

    The main method uses the complete condition-residual window. Setting
    ``use_fixed_low_pass=True`` activates the registered low-pass ablation; it
    is not part of the final deployed model.
    """

    def __init__(
        self,
        sensors: int = 14,
        window_size: int = 30,
        retained_bins: int = 5,
        use_fixed_low_pass: bool = False,
    ):
        super().__init__()
        self.use_fixed_low_pass = bool(use_fixed_low_pass)
        self.representation_mode = (
            "fixed_lowpass" if self.use_fixed_low_pass else "condition_residual"
        )
        self.low_pass = FixedLowPassProjector(window_size, retained_bins)
        self.temporal_encoder = MultiReceptiveFieldCausalEncoder(sensors)
        self.sensor_encoder = SensorSelfAttentionEncoder(window_size)

        protocol_weight, protocol_bias = _protocol_v1_fusion_initialization(sensors)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        with torch.no_grad():
            self.fusion[0].weight.copy_(protocol_weight)
            self.fusion[0].bias.copy_(protocol_bias)

    def encode(self, residual: Tensor) -> dict[str, Tensor]:
        model_input = self.low_pass(residual) if self.use_fixed_low_pass else residual
        temporal_features = self.temporal_encoder(model_input)
        sensor_features = self.sensor_encoder(model_input)
        fused = self.fusion(torch.cat([temporal_features, sensor_features], dim=1))
        return {
            "model_input": model_input,
            "temporal_features": temporal_features,
            "sensor_features": sensor_features,
            "fused_features": fused,
        }

    def forward(self, residual: Tensor) -> dict[str, Tensor]:
        encoded = self.encode(residual)
        prediction = self.regressor(encoded["fused_features"]).squeeze(-1)
        return {**encoded, "prediction": prediction}

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load current checkpoints or migrate registered pre-release keys."""

        if any(key.startswith("slow_encoder.") for key in state_dict):
            migrated: dict[str, Tensor] = {}
            for key, value in state_dict.items():
                if key.startswith("fast_encoder."):
                    continue
                if key.startswith("slow_encoder."):
                    key = "temporal_encoder." + key.removeprefix("slow_encoder.")
                if key == "fusion.0.weight" and value.shape[1] == 320:
                    value = value[:, :256]
                migrated[key] = value
            state_dict = migrated
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def freeze_source(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def build_conditioncomp_rul_net(
    *,
    sensors: int = 14,
    window_size: int = 30,
    retained_bins: int = 5,
    representation_mode: str = "condition_residual",
) -> ConditionCompRULNet:
    """Map frozen protocol labels to the final model or low-pass ablation."""

    aliases = {
        "condition_residual": "condition_residual",
        "fixed_lowpass": "fixed_lowpass",
        "raw_only": "condition_residual",
        "slow_only": "fixed_lowpass",
    }
    if representation_mode not in aliases:
        raise ValueError(
            "representation_mode must be condition_residual or fixed_lowpass."
        )
    resolved_mode = aliases[representation_mode]
    return ConditionCompRULNet(
        sensors=sensors,
        window_size=window_size,
        retained_bins=retained_bins,
        use_fixed_low_pass=resolved_mode == "fixed_lowpass",
    )
