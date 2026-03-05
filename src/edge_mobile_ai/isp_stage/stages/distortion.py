"""Stage 10 — Lens Distortion Correction."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np
from ..stage_base import ISPStage, StageConfig, StageResult


@dataclass
class DistortionConfig(StageConfig):
    k: list[float] = field(default_factory=lambda: [-0.1, 0.02, 0.0, 0.0, 0.0])
    fx: float | None = None   # focal length (pixels); auto if None
    fy: float | None = None
    cx: float | None = None   # principal point; auto-centred if None
    cy: float | None = None


'''
Sửa méo hình học do ống kính (barrel / pincushion / tangential distortion).
Khác với Lens Shading (sửa sáng), stage này sửa hình học không gian.

Stage này dùng để:
Sửa méo hình học do ống kính bằng cách áp dụng mô hình radial + tangential distortion và tái ánh xạ pixel.
Nó là bước xử lý hình học, không phải xử lý màu hay noise.
'''
class DistortionCorrectionStage(ISPStage):
    """Radial/tangential lens undistortion via OpenCV undistort."""

    name = "distortion"

    def __init__(
        self,
        enabled: bool = False,
        k: list[float] | None = None,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        **_: Any,
    ) -> None:
        super().__init__(
            DistortionConfig(
                enabled=enabled,
                k=k or [-0.1, 0.02, 0.0, 0.0, 0.0],
                fx=fx, fy=fy, cx=cx, cy=cy,
            )
        )


    def process(
            self,
            image: np.ndarray,
            metadata: dict[str, Any]
    ) -> StageResult:
        
        cfg: DistortionConfig = self.config # type: ignore[assignment]
        h, w = image.shape[:2]

        fx = cfg.fx or float(max(h, w))
        fy = cfg.fy or fx
        cx = cfg.cx or w / 2.0
        cy = cfg.cy or h / 2.0

        K = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=np.float64
        )
        dist = np.array(cfg.k[:5], dtype=np.float64)

        is_float = image.dtype.kind == "f"
        img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8) if is_float else image

        undistored = cv2.undistort(img8, K, dist)

        if is_float:
            out = undistored.astype(np.float32) / 255.0
        else:
            out = undistored

        metadata["distortion_coeffs"] = cfg.k
        return StageResult(
            image=out,
            metadata=metadata
        )