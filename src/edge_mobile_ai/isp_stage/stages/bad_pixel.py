"""Stage 3 — Bad Pixel Correction."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np
from ..stage_base import ISPStage, StageConfig, StageResult


@dataclass
class BadPixelConfig(StageConfig):
    threshold: float = 0.1   # relative deviation from neighbourhood median

'''
Bản chất vật lý
Sensor CMOS/CCD không bao giờ cho ra giá trị 0 tuyệt đối khi không có ánh sáng.
Nó luôn có:
Dark current
Read noise
ADC bias offset
Nên pixel tối nhất có thể là 64, 256, 1024… tùy sensor.

Nếu không trừ black level:
→ toàn bộ ảnh bị nâng nền lên
→ dynamic range bị sai

Black Level Correction:
Khôi phục đúng dynamic range vật lý
Làm shadow không bị xám
Là tiền đề cho white balance chính xác
Nếu làm sai bước này:
→ toàn bộ pipeline sau đó sai theo.
'''
class BadPixelCorrectionStage(ISPStage):
    """Detect and replace hot/dead pixels via median neighbourhood."""

    name = "bad_pixel"

    def __init__(
            self, 
            enabled: bool = True, 
            threshold: float = 0.1, 
            **_: Any
    ) -> None:
        super().__init__(BadPixelConfig(enabled=enabled, threshold=threshold))

    
    def process(
            self,
            image: np.ndarray,
            metadata: dict[str, Any]
    ) -> StageResult:
        
        cfg: BadPixelConfig = self.config # type: ignore[assignment]
        is_float = image.dtype.kind == "f"
        img = image.astype(np.float32)

        # Compute per-pixel median deviation
        med = cv2.medianBlur(
            (np.clip(img, 0, 1) * 255).astype(np.uint8) if is_float else img.astype(np.uint8),
            3
        ).astype(np.float32)

        if is_float:
            med /= 255.0

        diff = np.abs(img - med)
        if img.ndim == 3:
            bad_mask = diff.mean(axis=2) > cfg.threshold
        else:
            bad_mask = diff > cfg.threshold


        num_bad = int(bad_mask.sum())

        if num_bad > 0:
            if img.ndim == 3:
                for c in range(img.shape[2]):
                    img[:, :, c][bad_mask] = med[:, :, c][bad_mask]
            else:
                img[bad_mask] = med[bad_mask]

        metadata["bad_pixels_corrected"] = num_bad
        return StageResult(image=img, metadata=metadata)