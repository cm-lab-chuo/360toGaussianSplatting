"""
Stage 2 — Frame quality filtering (optional).

Scores each frame by Laplacian variance (a proxy for sharpness) and
removes low-quality frames before further processing.

Two modes (VideoSettings.filtermethod):
  "best_n"    — keep top N% or top N frames (framestokeep = "50%" or "100")
  "threshold" — keep frames whose sharpness score >= sharpnessthreshold

To replace with a different quality metric:
  Subclass FrameFilter, override `score_frame()`.
  Register the new class in registry.py under FRAME_FILTER.
"""
from __future__ import annotations
import logging
import shutil
from pathlib import Path

import cv2

from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class FrameFilter(Stage):

    @property
    def name(self) -> str:
        return "Frame Filter"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not self.cfg.video.useframefilter:
            logger.info("Frame filtering disabled — skipping.")
            return ctx

        assert ctx.frames_dir is not None, "frames_dir not set before FrameFilter"

        frames = sorted(ctx.frames_dir.glob("*.jpg")) + sorted(ctx.frames_dir.glob("*.png"))
        if not frames:
            logger.warning("No frames found in %s", ctx.frames_dir)
            return ctx

        logger.info("Scoring %d frames for sharpness…", len(frames))
        scored = [(f, self.score_frame(f)) for f in frames]

        keep = self._select(scored)
        remove = set(f for f, _ in scored) - set(keep)

        logger.info("Keeping %d / %d frames (removing %d)", len(keep), len(frames), len(remove))
        for f in remove:
            f.unlink()

        return ctx

    # ── sharpness metric ──────────────────────────────────────────────────

    def score_frame(self, path: Path) -> float:
        """Laplacian variance — higher means sharper."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    # ── selection logic ───────────────────────────────────────────────────

    def _select(self, scored: list[tuple[Path, float]]) -> list[Path]:
        v = self.cfg.video
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

        if v.filtermethod == "threshold":
            return [f for f, s in scored_sorted if s >= v.sharpnessthreshold]

        # best_n: parse framestokeep as percentage or absolute count
        keep_spec = v.framestokeep.strip()
        if keep_spec.endswith("%"):
            ratio = float(keep_spec[:-1]) / 100.0
            n = max(1, round(len(scored) * ratio))
        else:
            n = int(keep_spec)

        return [f for f, _ in scored_sorted[:n]]
