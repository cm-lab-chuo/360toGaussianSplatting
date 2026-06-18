"""
Stage 1 — Frame extraction from video using FFmpeg.

Extracts frames from an equirectangular 360° video at the configured rate.
Output frames are placed in ctx.frames_dir as JPEG files.

Configurable parameters (VideoSettings):
  fps                  — frames per second to extract
  frame_extraction_mode — "seconds" uses fps as interval; "frames" keeps every Nth frame
  resolutionwidth/height — output frame resolution
"""
from __future__ import annotations
import logging
from pathlib import Path

from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext
from utils.process import run

logger = logging.getLogger(__name__)


class FFmpegExtractor(Stage):

    def __init__(self, cfg: Config, ffmpeg_path: Path | None = None) -> None:
        super().__init__(cfg)
        self._ffmpeg = ffmpeg_path or self._resolve_ffmpeg(cfg)

    @staticmethod
    def _resolve_ffmpeg(cfg: Config) -> Path:
        # Use the configured path if provided ([ToolPaths] ffmpeg), else PATH.
        # An unset config value normalizes to Path("") == Path("."), so treat
        # "" and "." as "not configured".
        configured = cfg.tool_paths.ffmpeg
        if str(configured) not in ("", "."):
            return configured
        return Path("ffmpeg")

    # ── stage interface ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "Frame Extraction"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        out_dir = ctx.stage_dir("frames")
        v = self.cfg.video

        # Skip if frames already exist
        existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
        if existing:
            logger.info("Found %d existing frames, skipping extraction.", len(existing))
            ctx.frames_dir = out_dir
            return ctx

        if not ctx.input_path.exists():
            raise FileNotFoundError(f"Input not found: {ctx.input_path}")

        is_video = ctx.input_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".mts"}

        if is_video:
            self._extract_from_video(ctx.input_path, out_dir, v)
        else:
            # Input is a folder of images. COPY them into the work dir rather than
            # pointing frames_dir at the original — downstream stages (e.g. the
            # frame filter) delete rejected frames in place, which must never touch
            # the user's source data.
            self._copy_images(ctx.input_path, out_dir)
            frames = sorted(out_dir.glob("*.jpg")) + sorted(out_dir.glob("*.png"))
            logger.info("Copied %d images from folder → %s", len(frames), out_dir)
            ctx.frames_dir = out_dir
            return ctx

        frames = sorted(out_dir.glob("*.jpg"))
        logger.info("Extracted %d frames → %s", len(frames), out_dir)
        ctx.frames_dir = out_dir
        return ctx

    def _copy_images(self, src_dir: Path, out_dir: Path) -> None:
        import shutil
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in exts)
        if not images:
            raise RuntimeError(f"No images found in input folder: {src_dir}")
        for p in images:
            shutil.copy2(str(p), str(out_dir / p.name))

    def _extract_from_video(self, video: Path, out_dir: Path, v) -> None:
        out_pattern = str(out_dir / "%06d.jpg")

        vf_parts = []

        # Frame rate selection
        if v.frame_extraction_mode == "seconds":
            vf_parts.append(f"fps={v.fps}")
        else:
            # Every Nth frame (fps here is treated as 1/N)
            every_n = max(1, round(1.0 / v.fps))
            vf_parts.append(f"select='not(mod(n\\,{every_n}))',setpts=N/FRAME_RATE/TB")

        # Resolution scaling (preserve aspect ratio, pad to target if needed)
        vf_parts.append(
            f"scale={v.resolutionwidth}:{v.resolutionheight}"
            f":force_original_aspect_ratio=decrease"
            f",pad={v.resolutionwidth}:{v.resolutionheight}:(ow-iw)/2:(oh-ih)/2"
        )

        vf_filter = ",".join(vf_parts)

        cmd = [
            self._ffmpeg,
            "-i", video,
            "-vf", vf_filter,
            "-q:v", "2",         # JPEG quality (2 = near-lossless)
            "-vsync", "vfr",
            out_pattern,
        ]
        run(cmd)
