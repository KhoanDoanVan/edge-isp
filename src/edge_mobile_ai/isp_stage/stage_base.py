"""Abstract base class for all ISP pipeline stages."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class StageConfig:
    """Base configuration for every ISP stage."""

    enabled: bool = True
    debug: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Encapsulates the output of a single ISP stage."""

    image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class ISPStage(ABC):
    """
    Abstract base for every stage in the ISP pipeline.

    Subclasses must implement :meth:`process`.
    """

    #: Human-readable name shown in logs and reports.
    name: str = "unnamed_stage"
    
    def __init__(self, config: StageConfig | None = None) -> None:
        self.config = config or StageConfig()
        self._logger = logger.bind(stage=self.name)

    
    def __call__(self, image: np.ndarray, metadata: dict[str, Any] | None = None) -> StageResult:
        """Run the stage, respecting the *enabled* flag."""
        meta = metadata or {}
        if not self.config.enabled:
            self._logger.debug("Stage disabled — passthrough.")
            return StageResult(image=image, metadata=meta)
        
        import time

        t0 = time.perf_counter()
        result = self.process(image, meta)
        result.latency_ms = (time.perf_counter() - t0) * 1000.0

        self._logger.debug(f"Processed in {result.latency_ms:.2f} ms")

        if self.config.debug:
            self._debug_hook(result)

        return result
    

    @abstractmethod
    def process(self, image: np.ndarray, metadata: dict[str, Any]) -> StageResult:
        """
        Core processing logic.

        Parameters
        ----------
        image:
            Input image array. Dtype and value range depend on the stage's
            position in the pipeline (raw uint16, float32 [0,1], etc.).
        metadata:
            Mutable pipeline metadata dict (EXIF, calibration data, …).

        Returns
        -------
        StageResult
            Processed image and updated metadata.
        """

    
    def _debug_hook(self, result: StageResult) -> None:
        """Override for custom debug behaviour (save tiles, log stats, …)."""
        img = result.image
        self._logger.debug(
            f"[debug] shape={img.shape} dtype={img.dtype} "
            f"min={img.min():.4f} max={img.max():.4f} mean={img.mean():.4f}"
        )


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(enabled={self.config.enabled})"