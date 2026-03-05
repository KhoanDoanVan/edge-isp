"""
Stage 9 — Sharpening
=====================
Enhance image detail / edge contrast.

Methods
-------
* ``unsharp_mask``  — Classic unsharp mask (most widely used).
* ``laplacian``     — Laplacian high-frequency boost.
* ``high_pass``     — Frequency-domain high-pass injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from ..stage_base import ISPStage, StageConfig, StageResult


@dataclass
class SharpeningConfig(StageConfig):
    method: Literal["unsharp_mask", "laplacian", "high_pass"] = "unsharp_mask"
    amount: float = 0.5      # [0, 1] — sharpening intensity
    radius: float = 1.0      # blur radius for unsharp mask / high-pass
    threshold: float = 0.0   # suppress sharpening below this local contrast


'''
Tăng độ tương phản cạnh (edge contrast) và làm ảnh trông “nét” hơn sau khi đã qua denoise.

Trong pipeline trước đó bạn đã có:
Noise Reduction → làm mềm ảnh
NR luôn làm mất high-frequency detail.

Sharpening là bước:
Phục hồi hoặc tăng cường high-frequency

Nếu không có sharpening:
Ảnh nhìn “mềm”
Texture thiếu crisp
Viền không rõ


Sharpening không tăng detail thật
Nó chỉ tăng local contrast ở edge.
Mắt người hiểu local contrast là “nét”.

Stage này dùng để:
Tăng độ sắc nét cảm nhận bằng cách boost thành phần tần số cao của ảnh RGB.
Nó là bước “hoàn thiện thị giác” trước khi xuất ảnh.
'''
class SharpeningStage(ISPStage):
    """Edge & detail sharpening — operates on float32 [0, 1] RGB."""

    name = "sharpening"

    def __init__(
        self,
        enabled: bool = True,
        method: str = "unsharp_mask",
        amount: float = 0.5,
        radius: float = 1.0,
        threshold: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__(
            SharpeningConfig(
                enabled=enabled,
                method=method,  # type: ignore[arg-type]
                amount=amount,
                radius=radius,
                threshold=threshold,
            )
        )


    def process(
            self, 
            image: np.ndarray, 
            metadata: dict[str, Any]
    ) -> StageResult:
        
        cfg: SharpeningConfig = self.config
        amount = float(np.clip(cfg.amount, 0.0, 1.0))

        dispatch = {
            "unsharp_mask": self._unsharp_mask,
            "laplacian": self._laplacian,
            "high_pass": self._high_pass
        }

        fn = dispatch.get(cfg.method)
        if fn is None:
            raise ValueError(f"Unknown sharpening method: {cfg.method!r}")
        
        out = fn(image, amount, cfg.radius, cfg.threshold)
        np.clip(out, 0.0, 1.0, out=out)

        metadata["sharpen_method"] = cfg.method
        metadata["sharpen_amount"] = amount

        return StageResult(image=out, metadata=metadata)
    

    @staticmethod
    def _unsharp_mask(
        img: np.ndarray,
        amount: float,
        radius: float,
        threshold: float
    ) -> np.ndarray:
        """
        out = img + amount x (img - blur(img))
        """
        blurred = cv2.GaussianBlur(img, (0, 0), max(radius, 0.1))
        hf = img - blurred

        if threshold > 0.0:
            mask = (np.abs(hf).max(axis=2, keepdims=True) > threshold).astype(np.float32)
            hf = hf * mask

        sharpened = (img + amount * hf).astype(np.float32)
        return sharpened
    

    @staticmethod
    def _laplacian(
        img: np.ndarray,
        amount: float,
        radius: float,
        _threshold: float
    ) -> np.ndarray:
        img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lap = cv2.Laplacian(img8, cv2.CV_32F, ksize=3)
        lap_norm = lap / 255.0
        out = img + amount * 0.2 * lap_norm
        return out.astype(np.float32)
    

    @staticmethod
    def _high_pass(
        img: np.ndarray,
        amount: float,
        radius: float,
        _threshold: float
    ) -> np.ndarray:
        low = cv2.GaussianBlur(img, (0, 0), max(radius * 3, 1.0))
        high_pass = img - low + 0.5 # centred at 0.5
        # Blend: linear light composition
        out = img + amount * (high_pass - 0.5)
        return out.astype(np.float32)