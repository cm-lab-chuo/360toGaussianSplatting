"""
PipelineContext — shared state that flows through every stage.

Each stage reads from the context, does its work, writes its outputs
to disk, then updates the relevant path fields and returns the context.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class PipelineContext:
    # ── inputs ──────────────────────────────────────────────────────────
    input_path: Path       # original video or image folder
    work_dir: Path         # root working directory for this run

    # ── stage outputs (populated as stages complete) ─────────────────────
    frames_dir: Optional[Path] = None      # raw equirectangular frames
    cubemap_dir: Optional[Path] = None     # perspective crop images
    masked_dir: Optional[Path] = None      # masked versions of cubemap images
    mask_dir: Optional[Path] = None        # raw masks (any source) for SfM masking
    sparse_dir: Optional[Path] = None      # COLMAP sparse reconstruction

    # ── freeform metadata for stage-to-stage communication ───────────────
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── helpers ──────────────────────────────────────────────────────────
    def stage_dir(self, name: str) -> Path:
        """Return (and create) a subdirectory under work_dir for a stage."""
        d = self.work_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def images_for_sfm(self) -> Path:
        """
        Return the image directory that SfM should consume, preferring the most
        processed output available: masked > cubemap > raw frames.
        """
        def _has_files(d: Optional[Path]) -> bool:
            return bool(d) and d.exists() and any(d.iterdir())

        if _has_files(self.masked_dir):
            return self.masked_dir
        if _has_files(self.cubemap_dir):
            return self.cubemap_dir
        if _has_files(self.frames_dir):
            return self.frames_dir
        raise RuntimeError(
            "No images available for SfM — frames_dir, cubemap_dir and masked_dir "
            "are all empty or unset. Did an earlier stage fail?"
        )
