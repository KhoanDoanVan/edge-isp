"""
Stage 4 — Demosaicing
======================
Reconstruct a full-colour RGB image from a single-channel Bayer CFA image.

Supported methods
-----------------
* ``bilinear``  — Fast bilinear interpolation (via OpenCV).
* ``ahd``       — Adaptive Homogeneity-Directed (highest quality classic).
* ``vng``       — Variable Number of Gradients.
* ``edge_aware`` — Simple edge-aware / gradient-based demosaic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from ..stage_base import ISPStage, StageConfig, StageResult


BayerPattern = Literal["RGGB", "BGGR", "GRBG", "GBRG"]


@dataclass
class DemosaicConfig(StageConfig):
    method: Literal["bilinear", "ahd", "vng", "edge_aware"] = "ahd"
    bayer_pattern: BayerPattern = "RGGB"


# Map pattern string → OpenCV Bayer code (for 8/16-bit uint images)
_BAYER_CODE: dict[str, dict[str, int]] = {
    "bilinear": {
        "RGGB": cv2.COLOR_BAYER_RG2RGB,
        "BGGR": cv2.COLOR_BAYER_BG2RGB,
        "GRBG": cv2.COLOR_BAYER_GR2RGB,
        "GBRG": cv2.COLOR_BAYER_GB2RGB,
    },
    "vng": {
        "RGGB": cv2.COLOR_BAYER_RG2RGB_VNG,
        "BGGR": cv2.COLOR_BAYER_BG2RGB_VNG,
        "GRBG": cv2.COLOR_BAYER_GR2RGB_VNG,
        "GBRG": cv2.COLOR_BAYER_GB2RGB_VNG,
    },
    "ahd": {
        "RGGB": cv2.COLOR_BAYER_RG2RGB_EA,   # OpenCV uses EA as proxy; AHD via dcraw
        "BGGR": cv2.COLOR_BAYER_BG2RGB_EA,
        "GRBG": cv2.COLOR_BAYER_GR2RGB_EA,
        "GBRG": cv2.COLOR_BAYER_GB2RGB_EA,
    },
    "edge_aware": {
        "RGGB": cv2.COLOR_BAYER_RG2RGB_EA,
        "BGGR": cv2.COLOR_BAYER_BG2RGB_EA,
        "GRBG": cv2.COLOR_BAYER_GR2RGB_EA,
        "GBRG": cv2.COLOR_BAYER_GB2RGB_EA,
    },
}


class DemosaicStage(ISPStage):
    """Convert Bayer RAW → full-colour RGB."""

    name = "demosaic"

    def __init__(
        self,
        enabled: bool = True,
        method: str = "ahd",
        bayer_pattern: str = "RGGB",
        **_: Any,
    ) -> None:
        super().__init__(
            DemosaicConfig(
                enabled=enabled,
                method=method,  # type: ignore[arg-type]
                bayer_pattern=bayer_pattern,  # type: ignore[arg-type]
            )
        )


    def process(
            self,
            image: np.ndarray,
            metadata: dict[str, Any]
    ) -> StageResult:
        
        cfg: DemosaicConfig = self.config # type: ignore[assignment]

        if image.ndim != 2:
            raise ValueError(
                f"Demosaic expects a 2-D Bayer image, got shape {image.shape}."
            )
        
        # Convert float [0,1] -> uint16 for OpenCV
        is_float = image.dtype.kind == "f"

        if is_float:
            img16 = (np.clip(image, 0, 1) * 65535).astype(np.uint16)
        else:
            img16 = image.astype(np.uint16)

        code = _BAYER_CODE[cfg.method][cfg.bayer_pattern]
        rgb16 = cv2.cvtColor(img16, code)
        
        if is_float:
            rgb = rgb16.astype(np.float32) / 65535.0
        else:
            rgb = rgb16

        metadata["demosaic_method"] = cfg.method
        metadata["bayer_pattern"] = cfg.bayer_pattern

        return StageResult(image=rgb, metadata=metadata)