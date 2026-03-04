"""
Stage 5 — Noise Reduction
==========================
Spatial noise reduction with multiple algorithms.

Methods
-------
* ``gaussian``      — Fast Gaussian blur (baseline).
* ``bilateral``     — Edge-preserving bilateral filter.
* ``guided``        — Guided image filter (He et al., 2010). Best quality/speed.
* ``nlm``           — Non-local means (OpenCV FastNlMeans). Slow but thorough.
* ``median``        — Median filter. Excellent for impulse / salt-and-pepper noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from ..stage_base import ISPStage, StageConfig, StageResult


@dataclass
class NoiseReductionConfig(StageConfig):
    method: Literal["gaussian", "bilateral", "guided", "nlm", "median"] = "guided"
    strength: float = 0.5          # 0 = no NR, 1 = max NR
    preserve_edges: bool = True


'''
Giảm nhiễu (noise) trong ảnh sau khi đã xử lý RAW thành RGB float32.

Nếu không có Noise Reduction:
Ảnh ISO cao sẽ đầy hạt
Shadow bị bẩn
Skin nhìn rất xấu
Gradient bị lốm đốm

Camera của các hãng như:
Sony
Canon
Apple
đều có nhiều tầng Noise Reduction phức tạp hơn.
Code của bạn là phiên bản đơn khung (single-frame spatial NR).

Black Level & Lens Shading → sửa lỗi vật lý sensor
Noise Reduction → làm ảnh nhìn sạch hơn
Nó là bước “làm đẹp kỹ thuật” chứ không phải sửa vật lý.
'''
class NoiseReductionStage(ISPStage):
    """Apply spatial noise reduction to a float32 RGB image."""
    
    name = "noise_reduction"

    def __init__(
            self,
            enabled: bool = True,
            method: str = "guided",
            strength: float = 0.5,
            preserve_edges: bool = True,
            **_: Any
    ) -> None:
        super().__init__(
            NoiseReductionConfig(
                enabled=enabled,
                method=method,  # type: ignore[arg-type]
                strength=strength,
                preserve_edges=preserve_edges,
            )
        )


    def process(
            self,
            image: np.ndarray,
            metadata: dict[str, Any]
    ) -> StageResult:
        
        cfg: NoiseReductionConfig = self.config # type: ignore[assignment]
        s = float(np.clip(cfg.strength, 0.0, 1.0))

        dispatch = {
            "gaussian": self._gaussian,
            "bilateral": self._bilatera,
            "guided": self._guided,
            "nlm": self._nlm,
            "median": self._median
        }

        fn = dispatch.get(cfg.method)

        if fn is None:
            raise ValueError(f"Unknown NR method: {cfg.method!r}")
        
        denoised = fn(image, s)
        metadata["nr_method"] = cfg.method
        metadata["nr_strength"] = s

        return StageResult(image=denoised, metadata=metadata)


    @staticmethod
    def _gaussian(
        img: np.ndarray,
        strength: float
    ) -> np.ndarray:
        sigma = 0.5 + strength * 3.0
        return cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)
    

    @staticmethod
    def _bilatera(
        img: np.ndarray,
        strength: float
    ) -> np.ndarray:
        d = int(2 + strength * 12)
        sigma_color = 0.05 + strength * 0.15
        sigma_space = 2 + strength * 10
        img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        out8 = cv2.bilateralFilter(
            img8,
            d,
            sigma_color * 255,
            sigma_space
        )
        return out8.astype(np.float32) / 255.0
    

    @staticmethod
    def _guided(
        img: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Guided filter self-guided (image == guide)."""
        eps = (0.01 + strength * 0.1) ** 2
        r = int(4 + strength * 8)
        img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        out8 = cv2.ximgproc.guidedFilter(img8, img8, r, eps * 255**2) # type: ignore[attr-defined]
        return np.clip(out8.astype(np.float32) / 255.0, 0, 1)
    

    @staticmethod
    def _nlm(
        img: np.ndarray,
        strength: float
    ) -> np.ndarray:
        h_param = 3 + strength * 12
        img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img8.ndim == 3:
            out8 = cv2.fastNlMeansDenoisingColored(img8, None, h_param, h_param, 7, 21)
        else:
            out8 = cv2.fastNlMeansDenoising(img8, None, h_param, 7, 21)
        return out8.astype(np.float32) / 255.0
        

    @staticmethod
    def _median(
        img: np.ndarray,
        strength: float
    ) -> np.ndarray:
        ksize = int(3 + strength * 4) | 1 # ensure odd
        img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        out8 = cv2.medianBlur(img8, ksize)
        return out8.astype(np.float32) / 255.0