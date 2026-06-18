"""
Stage 4a — AutoMasker integration (GroundingDINO + SAM2).

Calls AutoMasker.exe to generate masks that exclude dynamic objects
(people, sky, vehicles, etc.) from the training images.

Configurable parameters (AutoMaskerSettings):
  keywords       — dot-separated detection targets, e.g. "person.sky.car"
  boxthreshold   — GroundingDINO box detection confidence
  textthreshold  — GroundingDINO text grounding threshold
  invertmask     — True means mask OUT the detected objects (keep background)
  maskexpand     — pixels to grow each mask region
  exportmaskonly — export binary mask PNGs alongside images

Output layout:
  ctx.masked_dir/
    images/      — masked JPEG images
    masks/       — binary mask PNGs (white = masked/excluded)

To swap with a different masker:
  Subclass Stage, implement run() with the same output convention,
  register in registry.py under MASKING.
"""
from __future__ import annotations
import logging
from pathlib import Path

from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext
from utils.process import run

logger = logging.getLogger(__name__)


class AutoMaskerStage(Stage):
    # NOTE: verify the flags below (--input/--text_prompt/--invert_mask/…) against
    # `AutoMasker.exe --help` before relying on this stage. See registry.py to
    # swap maskers.

    @property
    def name(self) -> str:
        return "AutoMasker"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        src_dir = ctx.cubemap_dir or ctx.frames_dir
        assert src_dir is not None, "No image directory available for masking"

        paths = self.cfg.automasker_paths
        settings = self.cfg.automasker

        exe = paths.automaskerpath
        if not exe.exists():
            raise FileNotFoundError(
                f"AutoMasker not found at {exe}\n"
                "Update [AutoMaskerPaths] automaskerpath in your config."
            )

        out_dir = ctx.stage_dir("masked")
        images_out = out_dir / "images"
        masks_out = out_dir / "masks"
        images_out.mkdir(exist_ok=True)
        masks_out.mkdir(exist_ok=True)

        keywords = settings.keywords.replace(".", " ")  # AutoMasker uses spaces

        cmd = [
            exe,
            "--input", src_dir,
            "--output", images_out,
            "--mask_output", masks_out,
            "--text_prompt", keywords,
            "--box_threshold", settings.boxthreshold,
            "--text_threshold", settings.textthreshold,
            "--mask_expand", settings.maskexpand,
        ]

        if settings.invertmask:
            cmd.append("--invert_mask")
        if settings.exporttransparent:
            cmd.append("--export_transparent")
        if paths.dinoconfig.exists():
            cmd += ["--dino_config", paths.dinoconfig]
        if paths.dinocheckpoint.exists():
            cmd += ["--dino_checkpoint", paths.dinocheckpoint]
        if paths.samconfig.exists():
            cmd += ["--sam_config", paths.samconfig]
        if paths.samcheckpoint.exists():
            cmd += ["--sam_checkpoint", paths.samcheckpoint]

        run(cmd)

        # Store the images subdirectory so SfM can find it directly
        ctx.masked_dir = images_out
        ctx.metadata["masks_dir"] = masks_out
        logger.info("Masked images → %s", images_out)
        return ctx
