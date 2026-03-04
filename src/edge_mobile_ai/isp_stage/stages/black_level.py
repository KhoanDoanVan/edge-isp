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

# Hiệu chỉnh mức độ đen
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


    def process(
            self, 
            image: np.ndarray, # raw image data (likely from a camera sensor)
            metadata: dict[str, Any]
    ) -> StageResult:

        cfg: BlackLevelConfig = self.config # type: ignore[assignment]

        # Raw images are often uint8, uint10, uint12, uint16 -> convert to float32
        img = image.astype(np.float32)

        # Black level = sensor bias (offset added by the camera even when no light hits the sensor).
        bl = cfg.black_level
        
        # Two cases: scalar vs per-channel black level
        if isinstance(bl, (list, tuple)):
            # per-channel (Bayer) black level
            # - creates a 2D map matching the Bayer pattern
            # This is sensor-accurate black level correction
            bl_map = self._build_bayer_map(bl, image.shape[:2])
            img -= bl_map
        else:
            # scalar black level
            # Meaning:
            # - Same black offset for every pixel
            # - Typical for grayscale or simplified pipelines
            img -= float(bl)

        # Clamp negatives to 0
        # because:
        # - After black subtraction, dark pixels may become negative
        # - Physically meaningless (negative light)
        np.clip(img, 0, None, out=img)

        # Optional normalization to [0, 1]
        if cfg.normalize:
            # white_level = maximum valid sensor value (e.g. 1023, 4095, 16383)
            # Dynamic range = white − black
            scale = cfg.white_level - (float(bl) if not isinstance(bl, (list, tuple)) else min(bl))
            # Avoid divide-by-zero
            img /= max(scale, 1.0)
            # Clamp to valid range
            np.clip(img, 0.0, 1.0, out=img)

        metadata["black_level_applied"] = bl
        metadata["white_level"] = cfg.white_level

        # new image with:
        # - black level substracted
        # - optionally normalized to [0,1]
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