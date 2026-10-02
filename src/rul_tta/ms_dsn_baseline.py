from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .protocol_v1_models import ConditionCompRULNet, build_conditioncomp_rul_net


class _AttentionStack(nn.Module):
    def __init__(self, width: int, heads: int = 4, layers: int = 3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, values: Tensor) -> Tensor:
        return self.norm(self.encoder(values))


class MSDSNRULModel(nn.Module):
    """Same-protocol implementation of the MS-DSN regression mechanism.

    The implementation preserves the published mechanism: a 100-dimensional
    pre-encoder, one shared three-layer/four-head attention encoder, one
    private encoder and decoder per source domain, and a two-layer shared
    predictor. Lightweight token-wise private branches keep this controlled
    four-epoch reimplementation computationally comparable. Only the shared
    path is used at target inference.
    """

    def __init__(
        self,
        sensors: int = 14,
        window_size: int = 30,
        source_domains: int = 3,
        width: int = 100,
        heads: int = 4,
        layers: int = 3,
        rul_scale: float = 125.0,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("MS-DSN width must be divisible by attention heads.")
        self.sensors = int(sensors)
        self.window_size = int(window_size)
        self.source_domains = int(source_domains)
        self.rul_scale = float(rul_scale)
        self.preencoder = nn.Sequential(
            nn.Linear(sensors, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, window_size, width))
        nn.init.normal_(self.position, std=0.02)
        self.shared_encoder = _AttentionStack(width, heads, layers)
        self.private_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Linear(width, width),
                    nn.LayerNorm(width),
                )
                for _ in range(source_domains)
            ]
        )
        self.private_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width * 2, width * 2),
                    nn.LayerNorm(width * 2),
                    nn.GELU(),
                    nn.Linear(width * 2, sensors),
                )
                for _ in range(source_domains)
            ]
        )
        self.shared_predictor = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def _shared(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.preencoder(values) + self.position
        shared_sequence = self.shared_encoder(encoded)
        shared_features = shared_sequence.mean(dim=1)
        normalized_rul = torch.sigmoid(
            self.shared_predictor(shared_features).squeeze(-1)
        )
        return encoded, shared_sequence, normalized_rul * self.rul_scale

    def forward(
        self,
        values: Tensor,
        domains: Tensor | None = None,
    ) -> dict[str, Tensor]:
        encoded, shared_sequence, prediction = self._shared(values)
        output = {
            "prediction": prediction,
            "fused_features": shared_sequence.mean(dim=1),
            "shared_sequence": shared_sequence,
        }
        if domains is None:
            return output
        private_sequence = torch.zeros_like(shared_sequence)
        reconstruction = torch.zeros_like(values)
        for domain_id in range(self.source_domains):
            mask = domains == domain_id
            if not bool(mask.any()):
                continue
            private = self.private_encoders[domain_id](encoded[mask])
            decoded = self.private_decoders[domain_id](
                torch.cat([shared_sequence[mask], private], dim=-1)
            )
            private_sequence[mask] = private
            reconstruction[mask] = decoded
        return {
            **output,
            "private_sequence": private_sequence,
            "reconstruction": reconstruction,
            "input": values,
        }


class MSDSNSourceOnlyRULModel(nn.Module):
    """Architecture-matched MS-DSN mechanism baseline.

    The shared predictor is the same condition-residual network used by the
    controlled ERM, CORAL and VREx baselines.
    Domain-private encoders, reconstruction, orthogonality, supervised
    label-similarity and DREx are training-only additions. This isolates the
    MS-DSN mechanism under the frozen four-epoch comparison budget.
    """

    def __init__(
        self,
        sensors: int = 14,
        window_size: int = 30,
        slow_bins: int = 5,
        source_domains: int = 3,
        rul_scale: float = 125.0,
        private_width: int = 32,
        representation_mode: str = "condition_residual",
    ):
        super().__init__()
        self.sensors = int(sensors)
        self.window_size = int(window_size)
        self.source_domains = int(source_domains)
        self.rul_scale = float(rul_scale)
        self.private_width = int(private_width)
        self.shared_model: ConditionCompRULNet = build_conditioncomp_rul_net(
            sensors=sensors,
            window_size=window_size,
            retained_bins=slow_bins,
            representation_mode=representation_mode,
        )
        flattened = sensors * window_size
        self.private_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(flattened, private_width),
                    nn.LayerNorm(private_width),
                    nn.GELU(),
                    nn.Linear(private_width, private_width),
                    nn.LayerNorm(private_width),
                )
                for _ in range(source_domains)
            ]
        )
        self.private_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128 + private_width, 64),
                    nn.LayerNorm(64),
                    nn.GELU(),
                    nn.Linear(64, flattened),
                )
                for _ in range(source_domains)
            ]
        )

    def _shared(self, values: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.shared_model.encode(values)
        fused = encoded["fused_features"]
        prediction = torch.sigmoid(
            self.shared_model.regressor(fused).squeeze(-1)
        ) * self.rul_scale
        return fused, prediction

    def forward(
        self,
        values: Tensor,
        domains: Tensor | None = None,
    ) -> dict[str, Tensor]:
        shared_features, prediction = self._shared(values)
        output = {
            "prediction": prediction,
            "fused_features": shared_features,
            "shared_sequence": shared_features[:, None, :],
        }
        if domains is None:
            return output
        flattened = values.flatten(1)
        private_features = shared_features.new_zeros(
            (len(values), self.private_width)
        )
        reconstruction = torch.zeros_like(flattened)
        for domain_id in range(self.source_domains):
            mask = domains == domain_id
            if not bool(mask.any()):
                continue
            private = self.private_encoders[domain_id](flattened[mask])
            decoded = self.private_decoders[domain_id](
                torch.cat([shared_features[mask], private], dim=1)
            )
            private_features[mask] = private
            reconstruction[mask] = decoded
        return {
            **output,
            "private_sequence": private_features[:, None, :],
            "reconstruction": reconstruction.reshape_as(values),
            "input": values,
        }


def _domain_statistics(
    errors: Tensor,
    domains: Tensor,
    psi: float,
) -> tuple[Tensor, Tensor]:
    risks: list[Tensor] = []
    boundary_biases: list[Tensor] = []
    for domain in torch.unique(domains, sorted=True):
        selected = errors[domains == domain]
        risks.append(selected.square().mean())
        # Equation (11) in Jia et al.; clamping protects the exponential
        # without changing the operating range of normalized RUL errors.
        positive = torch.exp((selected - psi).clamp(max=20.0))
        negative = torch.exp((-selected - psi).clamp(max=20.0))
        boundary_biases.append((positive + negative).mean())
    return torch.stack(risks), torch.stack(boundary_biases)


def _label_weighted_similarity(
    shared_features: Tensor,
    targets: Tensor,
    domains: Tensor,
    rul_scale: float,
    tau: float,
) -> Tensor:
    if len(shared_features) < 2:
        return shared_features.new_zeros(())
    standardized = (shared_features - shared_features.mean(dim=0)) / (
        shared_features.std(dim=0, unbiased=False).clamp_min(1e-5)
    )
    normalized = F.normalize(standardized, dim=1)
    cosine = normalized @ normalized.T
    label_distance = (targets[:, None] - targets[None, :]).abs() / rul_scale
    weights = torch.exp(-tau * label_distance)
    valid = ~torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    # Cross-domain pairs carry the domain-generalization signal. Same-domain
    # pairs remain available with half weight to preserve label geometry.
    cross_domain = domains[:, None] != domains[None, :]
    weights = weights * torch.where(cross_domain, 1.0, 0.5)
    weights = weights.masked_fill(~valid, 0.0)
    return (weights * (1.0 - cosine)).sum() / weights.sum().clamp_min(1e-5)


def _diversity_loss(
    shared_sequence: Tensor,
    private_sequence: Tensor,
    domains: Tensor,
) -> Tensor:
    losses: list[Tensor] = []
    for domain in torch.unique(domains, sorted=True):
        shared = shared_sequence[domains == domain].flatten(0, 1)
        private = private_sequence[domains == domain].flatten(0, 1)
        shared = shared - shared.mean(dim=0, keepdim=True)
        private = private - private.mean(dim=0, keepdim=True)
        cross_covariance = private.T @ shared / max(1, len(shared) - 1)
        losses.append(cross_covariance.square().mean())
    return torch.stack(losses).mean() if losses else shared_sequence.new_zeros(())


def ms_dsn_loss(
    output: dict[str, Tensor],
    targets: Tensor,
    domains: Tensor,
    rul_scale: float = 125.0,
    tau: float = 1.0,
    psi: float = 1.0,
    drex_weight: float = 1.0,
    task_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    diversity_weight: float = 1.0,
    similarity_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    errors = (output["prediction"] - targets) / rul_scale
    risks, boundary_biases = _domain_statistics(errors, domains, psi)
    drex = risks.var(unbiased=False) + boundary_biases.var(unbiased=False)
    task = risks.mean() + drex_weight * drex

    residual = output["input"] - output["reconstruction"]
    reconstruction = residual.square().mean() + residual.mean(dim=(1, 2)).square().mean()
    diversity = _diversity_loss(
        output["shared_sequence"], output["private_sequence"], domains
    )
    similarity = _label_weighted_similarity(
        output["fused_features"], targets, domains, rul_scale, tau
    )
    loss = (
        task_weight * task
        + reconstruction_weight * reconstruction
        + diversity_weight * diversity
        + similarity_weight * similarity
    )
    diagnostics = {
        "task": float(task.detach()),
        "drex": float(drex.detach()),
        "reconstruction": float(reconstruction.detach()),
        "diversity": float(diversity.detach()),
        "similarity": float(similarity.detach()),
    }
    return loss, diagnostics
