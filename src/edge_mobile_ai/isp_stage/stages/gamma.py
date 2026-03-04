"""
Stage 8 — Gamma & Tone Mapping
================================
Convert linear light values to a perceptually-encoded output.

Modes
-----
* ``srgb``          — Proper IEC 61966-2-1 sRGB piecewise EOTF.
* ``power``         — Simple gamma power law  out = in^(1/γ).
* ``reinhard``      — Global Reinhard tone mapping (for HDR inputs).
* ``filmic``        — ACES-inspired filmic S-curve.
* ``lut``           — Apply a 1-D LUT per channel (provide ``lut`` array).
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..stage_base import ISPStage, StageConfig, StageResult


@dataclass
class GammaToneConfig(StageConfig):
    mode: Literal["srgb", "power", "reinhard", "filmic", "lut"] = "srgb"
    gamma: float = 2.2             # used by 'power' mode
    lut: list[float] | None = None  # 256-element 1D LUT for 'lut' mode


# Biến đổi cách phân bố độ sáng (luminance) để phù hợp với hệ thị giác người.
class GammaToneMappingStage(ISPStage):
    """Apply gamma encoding / tone mapping to a linear float32 image."""

    name = "gamma_tone"

    def __init__(
        self,
        enabled: bool = True,
        mode: str = "srgb",
        gamma: float = 2.2,
        lut: list[float] | None = None,
        **_: Any,
    ) -> None:
        super().__init__(
            GammaToneConfig(
                enabled=enabled,
                mode=mode,  # type: ignore[arg-type]
                gamma=gamma,
                lut=lut,
            )
        )


    def process(
            self, 
            image: np.ndarray, 
            metadata: dict[str, Any]
    ) -> StageResult:

        cfg: GammaToneConfig = self.config # type: ignore[assignment]
        img = np.clip(image.astype(np.float32), 0.0, None)

        dispatch = {
            "srgb": self._srgb,
            "power": lambda x: self._power(x, cfg.gamma),
            "reinhard": self._reinhard,
            "filmic": self._filmic,
            "lut": lambda x: self._apply_lut(x, cfg.lut),
        }

        fn = dispatch.get(cfg.mode)
        if fn is None:
            raise ValueError(f"Unknown tone mode: {cfg.mode!r}")
        
        out = fn(img)
        np.clip(out, 0.0, 1.0, out=out)

        metadata["tone_mode"] = cfg.mode
        return StageResult(image=out,metadata=metadata)

    

    @staticmethod
    def _srgb(linear: np.ndarray) -> np.ndarray:
        """IEC 61966-2-1 sRGB EOTF (linearisation curve)."""
        out = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(np.maximum(linear, 0.0031308), 1.0 / 2.4) - 0.055,
        )
        return out.astype(np.float32)
    

    @staticmethod
    def _power(img: np.ndarray, gamma: float) -> np.ndarray:
        return np.power(np.maximum(img, 0.0), 1.0 / gamma).astype(np.float32)
    

    @staticmethod
    def _reinhard(img: np.ndarray) -> np.ndarray:
        """Global Reinhard: L_out = L_in / (1 + L_in)."""
        return (img / (1.0 + img)).astype(np.float32)
    

    @staticmethod
    def _filmic(img: np.ndarray) -> np.ndarray:
        """Simplified ACES-inspired filmic curve."""
        A, B, C, D, E, F = 0.22, 0.30, 0.10, 0.20, 0.01, 0.30
        num = img * (A * img + C * B) + D * E
        den = img * (A * img + B) + D * F
        return np.clip(num / np.maximum(den, 1e-8) - E / F, 0, 1).astype(np.float32)
    

    @staticmethod
    def _apply_lut(
        img: np.ndarray,
        lut: list[float] | None
    ) -> np.ndarray:
        if lut is None:
            raise ValueError("'lut' mode requires a non-No")
        lut_arr = np.array(lut, dtype=np.float32)
        n = len(lut_arr) - 1
        indices = np.clip(img * n, 0, n)
        lo = indices.astype(np.int32)
        hi = np.clip(lo + 1, 0, n)
        frac = indices - lo
        return (
            lut_arr[lo] * (1 - frac) + lut_arr[hi] * frac
        ).astype(np.float32)