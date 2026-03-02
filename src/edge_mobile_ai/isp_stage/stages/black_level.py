"""
Stage 1 — Black Level Correction
=================================
Subtract sensor black level (dark current / bias) from raw sensor data.
Supports per-channel Bayer black levels and optional clamping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..stage_base import ISPStage, StageConfig, StageResult


@dataclass
class BlackLevelConfig(StageConfig):
    """
    Parameters
    ----------
    black_level:
        Scalar or 4-element array [R, Gr, Gb, B] matching the Bayer pattern.
    white_level:
        Sensor white level (full well capacity). Used for normalisation.
    normalize:
        If True, output is float32 in [0, 1] after subtraction.
    """

    black_level: float | list[float] = 64.0
    white_level: float = 1023.0
    normalize: bool = True


class BlackLevelCorrectionStage(ISPStage):
    """Subtract sensor black level, optionally normalise to [0, 1]."""

    name = "black_level"

    def __init__(
            self,
            enabled: bool = True,
            black_level: float | list[float] = 64.0,
            white_level: float = 1023.0,
            normalize: bool = True,
            **_: Any
    ) -> None:
        super().__init__(
            BlackLevelConfig(
                enabled=enabled,
                black_level=black_level,
                white_level=white_level,
                normalize=normalize
            )
        )


    def process(self, image: np.ndarray, metadata: dict[str, Any]) -> StageResult:

        cfg: BlackLevelConfig = self.config # type: ignore[assignment]
        img = image.astype(np.float32)

        bl = cfg.black_level

        if isinstance(bl, (list, tuple)):
            bl_map = self._build_bayer_map(bl, image.shape[:2])
            img -= bl_map
        else:
            img -= float(bl)

        # Clamp negatives to 0
        np.clip(img, 0, None, out=img)

        if cfg.normalize:
            scale = cfg.white_level - (float(bl) if not isinstance(bl, (list, tuple)) else min(bl))
            img /= max(scale, 1.0)
            np.clip(img, 0.0, 1.0, out=img)

        metadata["black_level_applied"] = bl
        metadata["white_level"] = cfg.white_level

        return StageResult(image=img, metadata=metadata)


    @staticmethod
    def _build_bayer_map(
        channels: list[float],
        shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Build a per-pixel black-level map for Bayer RGGB layout:
            R  Gr
            Gb  B
        """

        if len(channels) != 4:
            raise ValueError("Per-channel black level must have exactly 4 values [R, Gr, Gb, B].")
        
        h, w = shape

        bl_map = np.zeros((h, w), dtype=np.float32)
        bl_map[0::2, 0::2] = channels[0] # R
        bl_map[0::2, 1::2] = channels[1] # Gr
        bl_map[1::2, 0::2] = channels[2] # Gb
        bl_map[1::2, 1::2] = channels[3] # B

        return bl_map