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

        # An unset config path normalizes to Path("") == Path("."), and "." passes
        # .exists() (it's the current dir), so check is_file() and reject empty.
        def _is_real_file(p: Path) -> bool:
            return str(p) not in ("", ".") and p.is_file()

        exe = paths.automaskerpath
        if not _is_real_file(exe):
            raise FileNotFoundError(
                f"AutoMasker executable not set or not found: '{exe}'\n"
                "Set [AutoMaskerPaths] automaskerpath in your config, or pass "
                "--config <path-to-default.ini>. (Did the config fail to load?)"
            )

        # AutoMasker writes everything (masked images and/or mask files) into a
        # single --output directory; there is no separate mask-output flag. Masks
        # get a suffix (default ".mask"); masked images keep the source name.
        out_dir = ctx.stage_dir("masked")

        # Keywords are passed verbatim. AutoMasker accepts dot- or comma-separated
        # targets (e.g. "person.sky" or "person,sky").
        keywords = settings.keywords

        cmd = [
            exe,
            "--input", src_dir,
            "--output", out_dir,
            "--keywords", keywords,
            "--box-threshold", settings.boxthreshold,
            "--text-threshold", settings.textthreshold,
            # AutoMasker requires an integer for --mask-expand (a float like "5.0"
            # is rejected with exit code 2).
            "--mask-expand", int(round(settings.maskexpand)),
        ]

        # Boolean export options (store_true flags).
        if settings.invertmask:
            cmd.append("--invert-mask")
        if settings.exporttransparent:
            cmd.append("--export-transparent")
        if settings.exportmaskonly:
            cmd.append("--export-mask-only")
        if settings.exportcolored:
            cmd.append("--export-jpg")

        # Optional custom mask to combine with AI-detected masks.
        if _is_real_file(settings.custom_mask_path):
            cmd += ["--custom-mask", settings.custom_mask_path]

        # Model paths (only pass when actually present; AutoMasker falls back to
        # its bundled Models folder otherwise).
        if _is_real_file(paths.dinoconfig):
            cmd += ["--dino-config", paths.dinoconfig]
        if _is_real_file(paths.dinocheckpoint):
            cmd += ["--dino-checkpoint", paths.dinocheckpoint]
        if _is_real_file(paths.samconfig):
            cmd += ["--sam-config", paths.samconfig]
        if _is_real_file(paths.samcheckpoint):
            cmd += ["--sam-checkpoint", paths.samcheckpoint]

        run(cmd)

        # AutoMasker writes into subfolders of --output, not the top level:
        #   <out>/mask_only/   binary mask PNGs   (preferred for COLMAP masking)
        #   <out>/transparent/ RGBA images (mask in alpha)
        # Point mask_dir at the binary masks if present, falling back sensibly.
        mask_src = out_dir
        for cand in ("mask_only", "masks", "transparent"):
            d = out_dir / cand
            if d.is_dir() and any(d.glob("*.png")):
                mask_src = d
                break

        # Handoff to SfM via COLMAP's mask_path mechanism: keep the original
        # (cubemap) images as the image set and expose the mask dir. The SfM stage
        # normalizes these into COLMAP's convention and applies them. We do NOT set
        # masked_dir, so images and masks never get mixed in one folder.
        ctx.mask_dir = mask_src
        logger.info("AutoMasker output → %s (masks will be applied in SfM)", mask_src)
        return ctx
