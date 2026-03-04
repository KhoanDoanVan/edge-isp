"""
Stage 7 — Color Correction Matrix (CCM)
========================================
Apply a 3×3 colour correction matrix to map camera-native RGB to
a target colour space (sRGB, DCI-P3, etc.).

The CCM is typically derived via calibration against a colour checker
(see ``calibration/ccm_solver.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..stage_base import ISPStage, StageConfig, StageResult


# ITU-R BT.709 / sRGB D65 — identity-ish as a sensible default
_IDENTITY_CCM: list[list[float]] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

# Typical camera-to-sRGB CCM (approximate, real values come from calibration)
_D65_CCM: list[list[float]] = [
    [ 1.964,  -0.798,  -0.167],
    [-0.201,   1.520,  -0.319],
    [ 0.012,  -0.438,   1.426],
]


@dataclass
class ColorCorrectionConfig(StageConfig):
    ccm: list[list[float]] = field(default_factory=lambda: _D65_CCM)
    illuminant: str = "D65"
    clip_output: bool = True

'''
Camera sensor không nhìn màu giống mắt người.
Mỗi sensor có:
Spectral response khác nhau
Bộ filter Bayer khác nhau
Crosstalk giữa channel
Vì vậy RGB từ sensor là:
Camera-native RGB, không phải sRGB.

nếu gamma làm “độ sáng nhìn tự nhiên”, thì CCM làm “màu nhìn đúng”.
'''
class ColorCorrectionStage(ISPStage):
    """Apply 3x3 CCM to convert camera native RGB → target colour space."""

    name = "color_correction"

    def __init__(
            self,
            enabled: bool = True,
            ccm: list[list[float]] | None = None,
            illuminant: str = "D65",
            clip_output: bool = True,
            **_: Any
    ) -> None:
        super().__init__(
            ColorCorrectionConfig(
                enabled=enabled,
                ccm=ccm if ccm is not None else _D65_CCM,
                illuminant=illuminant,
                clip_output=clip_output
            )
        )


    def process(
            self,
            image: np.ndarray,
            metadata: dict[str, Any]
    ) -> StageResult:
        
        cfg: ColorCorrectionConfig = self.config # type: ignore[assignment]
        M = np.array(cfg.ccm, dtype=np.float32) # 3x3

        if M.shape != (3, 3):
            raise ValueError(f"CCM must be 3x3, got {M.shape}")
        
        h, w, c = image.shape
        flat = image.reshape(-1, 3) # (N, 3)
        corrected = flat @ M.T # (N, 3) x (3x3) -> (N, 3)
        out = corrected.reshape(h, w, 3).astype(np.float32)

        if cfg.clip_output:
            np.clip(out, 0.0, 1.0, out=out)

        metadata["ccm"] = M.tolist()
        metadata["illuminant"] = cfg.illuminant
        return StageResult(image=out, metadata=metadata)