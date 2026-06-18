"""
Stage 4c — Use pre-generated masks.

Copies masks from a user-specified folder and symlinks/copies images so
that masked_dir has the right layout for SfM.

Expected mask naming: same stem as the source image, .png extension.
E.g. 000001_y000_p+000.jpg → 000001_y000_p+000.png
"""
from __future__ import annotations
import logging
import shutil
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
        if not src_masks or not Path(src_masks).exists():
            raise FileNotFoundError(
                f"pregenerated_masks_path not set or does not exist: {src_masks}\n"
                "Update [AutoMaskerSettings] pregenerated_masks_path in your config."
            )

        src_dir = ctx.cubemap_dir or ctx.frames_dir
        assert src_dir is not None

        out_dir = ctx.stage_dir("masked")
        images_out = out_dir / "images"
        masks_out = out_dir / "masks"
        images_out.mkdir(exist_ok=True)
        masks_out.mkdir(exist_ok=True)

        images = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")))
        copied, missing = 0, 0
        for img_path in images:
            mask_path = Path(src_masks) / (img_path.stem + ".png")
            if mask_path.exists():
                shutil.copy2(str(img_path), str(images_out / img_path.name))
                shutil.copy2(str(mask_path), str(masks_out / mask_path.name))
                copied += 1
            else:
                logger.warning("No mask for %s — skipping image.", img_path.name)
                missing += 1

        logger.info("Copied %d image+mask pairs (%d missing)", copied, missing)
        ctx.masked_dir = images_out
        ctx.metadata["masks_dir"] = masks_out
        return ctx
