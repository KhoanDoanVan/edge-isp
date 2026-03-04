"""
Stage 6 — White Balance
========================
Scale R / G / B channels so the scene appears neutral.

Methods
-------
* ``grey_world``        — Assumes average scene reflectance is achromatic.
* ``perfect_reflector`` — Normalises to the brightest (99th-percentile) patch.
* ``manual``            — Explicit gains [r_gain, g_gain, b_gain].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..stage_base import ISPStage, StageConfig, StageResult

@dataclass
class WhiteBalanceConfig(StageConfig):
    method: Literal["grey_world", "perfect_reflector", "manual"] = "grey_world"
    gains: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    percentile: float = 99.0   # used by perfect_reflector

'''
Sensor ghi nhận đúng vật lý:
Vật trắng dưới đèn vàng → pixel vàng.
Nhưng mắt người có cơ chế chromatic adaptation nên ta vẫn thấy nó trắng.


Nếu CCM sửa “màu camera”,
thì WB sửa “màu của nguồn sáng”.
'''
class WhiteBalanceStage(ISPStage):
    """Per-channel gain-based white balance correction."""

    name = "white_balance"

    def __init__(
            self, 
            enabled: bool = True,
            method: str = "grey_world",
            gains: list[float] | None = None,
            percentile: float = 99.0,
            **_: Any
    ) -> None:
        super().__init__(
            WhiteBalanceConfig(
                enabled=enabled,
                method=method, # type: ignore[arg-type]
                gains=gains or [1.0, 1.0, 1.0],
                percentile=percentile
            )
        )


    def process(
            self,
            image: np.ndarray,
            metadata: dict[str, Any]
    ) -> StageResult:
        
        cfg: WhiteBalanceConfig = self.config # type: ignore[assignment]

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("White balance expects an HxWx3 float32 RGB image.")
        
        if cfg.method == "grey_world":
            gains = self._grey_world_gains(image)
        elif cfg.method == "perfect_reflector":
            gains = self._perfect_reflector_gains(image, cfg.percentile)
        elif cfg.method == "manual":
            gains = np.array(cfg.gains, dtype=np.float32)
        else:
            raise ValueError(f"Unknown WB method: {cfg.method!r}")
        
        out = np.clip(image * gains[np.newaxis, np.newaxis, :], 0.0, 1.0).astype(np.float32)
        metadata["wb_gains"] = gains.tolist()
        metadata["wb_method"] = cfg.method

        return StageResult(image=out, metadata=metadata)
        

        
    
    @staticmethod
    def _grey_world_gains(img: np.ndarray) -> np.ndarray:
        means = img.mean(axis=(0, 1)) # [r_mean, g_mean, b_mean]
        g_ref = means[1] if means[1] > 1e-6 else 1.0
        gains = g_ref / np.clip(means, 1e-6, None)
        return gains.astype(np.float32)


    @staticmethod
    def _perfect_reflector_gains(
        img: np.ndarray,
        percentile: float
    ) -> np.ndarray:
        peaks = np.percentile(img.reshape(-1, 3), percentile, axis=0)
        ref = max(peaks.max(), 1e-6)
        gains = ref / np.clip(peaks, 1e-6, None)
        return gains.astype(np.float32)