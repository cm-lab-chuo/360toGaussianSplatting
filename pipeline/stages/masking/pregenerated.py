"""
Stage 4c — Use pre-generated masks (e.g. computed by a separate program).

Points the pipeline at an existing folder of masks. The SfM stage normalizes
them into COLMAP's mask convention and applies them, so any naming/polarity the
external tool used is handled there (see utils/masks.py and [MaskSettings]).

This masker does not modify or copy anything — it only records the mask source.
The mask folder is taken from [AutoMaskerSettings] pregenerated_masks_path.
"""
from __future__ import annotations
import logging
from pathlib import Path

from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class PregeneratedMasker(Stage):

    @property
    def name(self) -> str:
        return "Masking (pre-generated)"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        src_masks = self.cfg.automasker.pregenerated_masks_path
        if not src_masks or str(src_masks) in ("", ".") or not Path(src_masks).exists():
            raise FileNotFoundError(
                f"pregenerated_masks_path not set or does not exist: {src_masks}\n"
                "Set [AutoMaskerSettings] pregenerated_masks_path to the folder of "
                "masks produced by your external program."
            )

        ctx.mask_dir = Path(src_masks)
        logger.info("Using pre-generated masks → %s (applied in SfM)", ctx.mask_dir)
        return ctx
