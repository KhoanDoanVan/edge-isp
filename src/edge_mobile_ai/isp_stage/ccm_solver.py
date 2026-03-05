"""
CCM Solver — Least-Squares Color Correction Matrix Calibration
==============================================================
Fit a 3x3 CCM given a set of measured camera RGB patches and
their reference sRGB targets (e.g. from a 24-patch ColorChecker).

Usage
-----
>>> solver = CCMSolver()
>>> ccm = solver.fit(camera_patches, reference_patches)
>>> corrected = solver.apply(raw_image, ccm)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class CCMSolver:
    """Least-squares 3x3 CCM estimator."""

    def __init__(self, regularisation: float = 1e-4) -> None:
        self.regularisation = regularisation
        self.ccm_: np.ndarray | None = None

    # ------------------------------------------------------------------

    def fit(
        self,
        camera: np.ndarray,   # (N, 3) measured camera RGB [0,1]
        reference: np.ndarray,  # (N, 3) ground-truth target RGB [0,1]
    ) -> np.ndarray:
        """
        Solve CCM via ridge-regularised least squares:
            min ||camera @ CCM.T - reference||^2 + λ ||CCM||^2

        Returns
        -------
        ccm : ndarray, shape (3, 3)
        """
        if camera.shape != reference.shape or camera.ndim != 2 or camera.shape[1] != 3:
            raise ValueError("camera and reference must both be (N, 3) arrays.")

        n = camera.shape[0]
        # Build normal equations: (A^T A + λI) x = A^T b
        A = camera   # (N, 3)
        B = reference  # (N, 3)
        ATA = A.T @ A + self.regularisation * np.eye(3, dtype=np.float64)
        ATB = A.T @ B
        ccm = np.linalg.solve(ATA, ATB).T  # (3, 3)
        self.ccm_ = ccm.astype(np.float32)
        return self.ccm_

    def apply(self, image: np.ndarray, ccm: np.ndarray | None = None) -> np.ndarray:
        """Apply CCM to an (H, W, 3) float32 image."""
        M = ccm if ccm is not None else self.ccm_
        if M is None:
            raise RuntimeError("No CCM available. Call fit() first or pass ccm=.")
        h, w, _ = image.shape
        out = (image.reshape(-1, 3) @ M.T).reshape(h, w, 3)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def deltaE_2000(
        self, camera: np.ndarray, reference: np.ndarray, ccm: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Compute ΔE 2000 statistics (requires sklearn for Lab conversion)."""
        try:
            from skimage.color import deltaE_ciede2000, rgb2lab
        except ImportError:
            return {"error": "scikit-image not installed. pip install scikit-image"}

        corrected = self.apply(camera.reshape(1, -1, 3), ccm).reshape(-1, 3)
        lab_pred = rgb2lab(corrected.reshape(1, -1, 3)).reshape(-1, 3)
        lab_ref = rgb2lab(reference.reshape(1, -1, 3)).reshape(-1, 3)
        de = deltaE_ciede2000(lab_ref, lab_pred)
        return {"mean": float(de.mean()), "max": float(de.max()), "per_patch": de.tolist()}

    def save(self, path: str | Path) -> None:
        if self.ccm_ is None:
            raise RuntimeError("No CCM to save.")
        np.save(str(path), self.ccm_)

    def load(self, path: str | Path) -> np.ndarray:
        self.ccm_ = np.load(str(path))
        return self.ccm_