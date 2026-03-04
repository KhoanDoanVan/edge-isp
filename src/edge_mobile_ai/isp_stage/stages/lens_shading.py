"""Stage 2 — Lens Shading Correction (vignetting compensation)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..stage_base import ISPStage, StageResult, StageConfig



@dataclass
class LensShadingConfig(StageConfig):
    gain_map: list[list[float]] | None = None # HxW float32 gain map
    fallback_order: int = 4 # polynomial order for synthetic flat-field


'''
sửa hiện tượng vignetting (rìa ảnh tối hơn trung tâm).

Do:
Hình học thấu kính
Góc tới của ánh sáng
Che khuất cơ học
Microlens inefficiency
→ Pixel ở rìa nhận ít photon hơn.
Kết quả:
Trung tâm sáng
Viền tối

Nếu WB sửa “màu ánh sáng”,
CCM sửa “màu camera”,
thì LSC sửa “độ sáng theo vị trí”.
'''
class LensShadingCorrectionStage(ISPStage):
    """Multiply by a gain map to compensate lens vignetting."""

    name = "lens_shading"

    def __init__(
            self, 
            enabled: bool = True,
            gain_map: list[list[float]] | None = None,
            fallback_order: int = 4,
            **_: Any
    ) -> None:
        super().__init__(LensShadingConfig(
            enabled=enabled,
            gain_map=gain_map,
            fallback_order=fallback_order
        ))


    def process(self, image: np.ndarray, metadata: dict[str, Any]) -> StageResult:

        cfg: LensShadingConfig = self.config
        h, w = image.shape[:2]

        # real calibration mode
        if cfg.gain_map is not None:
            gain = np.array(cfg.gain_map, dtype=np.float32)
            import cv2
            gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            gain = self._synthetic_gain_map(h, w, cfg)
            
        if image.ndim == 3:
            gain = gain[:, :, np.newaxis]

        out = (image * gain).astype(np.float32)
        np.clip(out, 0.0, 1.0, out=out)
        metadata["lens_shading_applied"] = True
        return StageResult(image=out, metadata=metadata)


    @staticmethod
    def _synthetic_gain_map(h: int, w: int, order: int) -> np.ndarray:
        """Radial cosine^order falloff — placeholder if no real map is provided."""

        cx, cy = w / 2.0, h / 2.0
        y, x = np.mgrid[:h, :w]
        r = np.sqrt(((x - cx) / cx)**2 + ((y - cy) / cy)**2)
        gain = 1.0 / (np.cos(np.clip(r, 0, 1) * np.pi / 2)**order + 1e-8)
        gain = gain / gain[int(cy), int(cx)]
        return gain.astype(np.float32)