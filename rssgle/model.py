from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .frs import AffineResidualHead
from .sgle import SGLEConfig, sgle_refine


class SpectralProjection(nn.Module):
    """Three BN-1x1Conv-LeakyReLU projections used by the paper."""

    def __init__(self, bands: int, width: int = 128) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(3):
            in_channels = bands if index == 0 else width
            layers.extend(
                [
                    nn.BatchNorm2d(in_channels),
                    nn.Conv2d(in_channels, width, kernel_size=1),
                    nn.LeakyReLU(inplace=False),
                ]
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class MSSSBlock(nn.Module):
    """Multi-scale spectral-spatial block in its deployment form."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: tuple[int, ...] = (5, 7, 9),
    ) -> None:
        super().__init__()
        if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("kernels must be non-empty positive odd integers")
        self.norm = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.activation = nn.LeakyReLU(inplace=False)
        self.depthwise = nn.ModuleList(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=out_channels,
                bias=False,
            )
            for kernel in kernels
        )
        self.fusion = nn.Conv2d(
            len(kernels) * out_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.fusion.weight, a=0.01)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        feature = self.activation(self.pointwise(self.norm(value)))
        responses = [convolution(feature) for convolution in self.depthwise]
        return self.activation(self.fusion(torch.cat(responses, dim=1)))


class MSSSBackbone(nn.Module):
    """Full-scene MSSS feature extractor and base classifier."""

    def __init__(
        self,
        bands: int,
        classes: int,
        widths: tuple[int, int] = (128, 64),
        kernels: tuple[int, ...] = (5, 7, 9),
    ) -> None:
        super().__init__()
        self.bands = bands
        self.classes = classes
        self.feature_dim = widths[-1]
        self.spectral_projection = SpectralProjection(bands, width=128)
        self.blocks = nn.ModuleList(
            [
                MSSSBlock(128, widths[0], kernels),
                MSSSBlock(widths[0], widths[1], kernels),
            ]
        )
        self.classifier = nn.Linear(widths[-1], classes)

    def features(self, cube_hwb: torch.Tensor) -> torch.Tensor:
        if cube_hwb.ndim != 3 or cube_hwb.shape[-1] != self.bands:
            raise ValueError(
                f"expected [H,W,{self.bands}], got {tuple(cube_hwb.shape)}"
            )
        feature = cube_hwb.permute(2, 0, 1).unsqueeze(0).float()
        feature = self.spectral_projection(feature)
        for block in self.blocks:
            feature = block(feature)
        return feature.squeeze(0).permute(1, 2, 0).contiguous()

    def forward(self, cube_hwb: torch.Tensor) -> torch.Tensor:
        feature = self.features(cube_hwb)
        return self.classifier(feature)


@dataclass(frozen=True)
class RSSGLEConfig:
    bands: int
    classes: int
    radius: int = 3
    affinity_temperature: float = 1.0
    message_strength: float = 1.5


class RSSGLE(nn.Module):
    """Deployment graph: MSSS + folded FRS residual + parameter-free SGLE."""

    def __init__(self, config: RSSGLEConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = MSSSBackbone(config.bands, config.classes)
        self.frs = AffineResidualHead(self.backbone.feature_dim, config.classes)

    def pre_sgle_logits(self, cube_hwb: torch.Tensor) -> torch.Tensor:
        feature = self.backbone.features(cube_hwb)
        return self.backbone.classifier(feature) + self.frs(feature)

    def forward(self, cube_hwb: torch.Tensor) -> torch.Tensor:
        logits = self.pre_sgle_logits(cube_hwb)
        return sgle_refine(
            cube_hwb,
            logits,
            SGLEConfig(
                radius=self.config.radius,
                affinity_temperature=self.config.affinity_temperature,
                message_strength=self.config.message_strength,
            ),
        )
