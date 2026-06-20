"""
Stage 3 — Equirectangular → perspective crop splitting.

For each equirectangular frame, generates `splits` perspective views evenly
distributed around the horizontal ring (plus optional tilt rows).
This converts one 360° frame into multiple standard-perspective images that
SfM tools like COLMAP/SphereSFM can process with a known camera model.

Layout with splits=8, usemultitilt=False:
  8 views at yaw = 0°, 45°, 90°, … 315°, pitch = 0°

Layout with splits=8, usemultitilt=True, tiltangle2=30, tiltangle3=-30:
  24 views: 8 at pitch=0°, 8 at pitch=+30°, 8 at pitch=-30°

Output naming convention:
  {frame_index:06d}_y{yaw:03d}_p{pitch:+04d}.jpg

Configurable parameters (VideoSettings):
  splits         — number of horizontal views
  fovvalue       — horizontal/vertical FOV of each perspective crop (degrees)
  resolutionwidth/height — output crop resolution
  usemultitilt   — add tilt rows
  tiltangle2/3   — tilt angles (degrees)

To replace the projection algorithm:
  Subclass CubemapSplitter, override `reproject()`.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class CubemapSplitter(Stage):
    _MANIFEST_NAME = ".cubemap-manifest.json"

    @property
    def name(self) -> str:
        return "Cubemap Split"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.frames_dir is not None, "frames_dir not set before CubemapSplitter"

        out_dir = ctx.stage_dir("cubemap")
        frames = sorted(
            list(ctx.frames_dir.glob("*.jpg")) + list(ctx.frames_dir.glob("*.png"))
        )
        if not frames:
            raise RuntimeError(f"No frames in {ctx.frames_dir}")

        # Build the list of (yaw, pitch) view directions
        view_angles = list(self._view_angles())
        expected = len(frames) * len(view_angles)
        expected_names = self._expected_names(frames, view_angles)
        manifest = self._build_manifest(frames, expected_names)
        manifest_path = out_dir / self._MANIFEST_NAME

        # Idempotent resume: if a previous run already produced the full set of
        # crops, skip regeneration (this stage is the slow part of the pipeline).
        existing = len(list(out_dir.glob("*.jpg")))
        if self._cache_is_valid(out_dir, manifest_path, manifest, expected_names):
            logger.info("Found %d existing crops (>= %d expected) — skipping split.",
                        existing, expected)
            ctx.cubemap_dir = out_dir
            return ctx

        for stale in out_dir.glob("*.jpg"):
            stale.unlink()
        manifest_path.unlink(missing_ok=True)

        logger.info(
            "Splitting %d frames × %d views = %d images",
            len(frames), len(view_angles), expected,
        )

        v = self.cfg.video
        for seq, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            if img is None:
                logger.warning("Could not read %s — skipping", frame_path)
                continue

            # Use the sorted position as the frame index so output names are
            # deterministic and collision-free regardless of source naming.
            idx = int(frame_path.stem) if frame_path.stem.isdigit() else seq
            for yaw, pitch in view_angles:
                crop = self.reproject(img, yaw, pitch, v.fovvalue, v.resolutionwidth, v.resolutionheight)
                out_name = f"{idx:06d}_y{yaw:03d}_p{pitch:+04d}.jpg"
                if not cv2.imwrite(
                    str(out_dir / out_name),
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise RuntimeError(
                        f"Failed to write cubemap crop: {out_dir / out_name}"
                    )

        total = len(list(out_dir.glob("*.jpg")))
        if total != expected:
            raise RuntimeError(
                f"Cubemap generation incomplete: expected {expected} files, got {total}"
            )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Generated %d perspective crops → %s", total, out_dir)
        ctx.cubemap_dir = out_dir
        return ctx

    # ── view angle generator ──────────────────────────────────────────────

    def _expected_names(
        self,
        frames: list[Path],
        view_angles: list[tuple[int, int]],
    ) -> list[str]:
        names = []
        for seq, frame_path in enumerate(frames):
            idx = int(frame_path.stem) if frame_path.stem.isdigit() else seq
            names.extend(
                f"{idx:06d}_y{yaw:03d}_p{pitch:+04d}.jpg"
                for yaw, pitch in view_angles
            )
        if len(names) != len(set(names)):
            raise RuntimeError(
                "Input frame names produce duplicate cubemap output names. "
                "Rename duplicate numeric stems before running the splitter."
            )
        return names

    def _build_manifest(
        self,
        frames: list[Path],
        expected_names: list[str],
    ) -> dict:
        v = self.cfg.video
        return {
            "version": 1,
            "inputs": [
                {
                    "name": frame.name,
                    "size": frame.stat().st_size,
                    "mtime_ns": frame.stat().st_mtime_ns,
                }
                for frame in frames
            ],
            "settings": {
                "splits": v.splits,
                "fovvalue": v.fovvalue,
                "resolutionwidth": v.resolutionwidth,
                "resolutionheight": v.resolutionheight,
                "usemultitilt": v.usemultitilt,
                "tiltangle2": v.tiltangle2,
                "tiltangle3": v.tiltangle3,
            },
            "outputs": expected_names,
        }

    @staticmethod
    def _cache_is_valid(
        out_dir: Path,
        manifest_path: Path,
        manifest: dict,
        expected_names: list[str],
    ) -> bool:
        if not manifest_path.is_file():
            return False
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if cached != manifest:
            return False
        existing_names = {path.name for path in out_dir.glob("*.jpg")}
        return existing_names == set(expected_names)

    def _view_angles(self) -> Iterable[tuple[int, int]]:
        v = self.cfg.video
        step = 360 // v.splits
        pitches = [0]
        if v.usemultitilt:
            pitches += [int(v.tiltangle2), int(v.tiltangle3)]
        for pitch in pitches:
            for i in range(v.splits):
                yield (i * step, pitch)

    # ── projection ───────────────────────────────────────────────────────

    def reproject(
        self,
        equirect: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        out_w: int,
        out_h: int,
    ) -> np.ndarray:
        """
        Extract a rectilinear perspective crop from an equirectangular image.

        Convention: yaw rotates around the vertical (Y) axis (left/right),
        pitch tilts up/down.  Positive yaw looks to the right; positive pitch
        looks up.
        """
        h_in, w_in = equirect.shape[:2]

        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)

        # Focal length in pixels for the desired FOV
        f = out_w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))

        # Pixel grid for the output image (camera space)
        xs = np.linspace(-out_w / 2.0, out_w / 2.0, out_w, dtype=np.float32)
        ys = np.linspace(-out_h / 2.0, out_h / 2.0, out_h, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)

        # Direction vectors in camera space (z forward)
        zg = np.full_like(xg, f)
        norm = np.sqrt(xg ** 2 + yg ** 2 + zg ** 2)
        xg, yg, zg = xg / norm, yg / norm, zg / norm

        # Rotate: pitch (around X axis) then yaw (around Y axis)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # Pitch rotation
        xr = xg
        yr = cos_p * yg - sin_p * zg
        zr = sin_p * yg + cos_p * zg

        # Yaw rotation
        xw = cos_y * xr + sin_y * zr
        yw = yr
        zw = -sin_y * xr + cos_y * zr

        # Spherical → equirectangular UV
        lon = np.arctan2(xw, zw)            # −π … π
        lat = np.arcsin(np.clip(yw, -1, 1)) # −π/2 … π/2

        map_x = ((lon / np.pi + 1.0) / 2.0 * w_in).astype(np.float32)
        map_y = ((-lat / (np.pi / 2.0) + 1.0) / 2.0 * h_in).astype(np.float32)

        crop = cv2.remap(
            equirect, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        return crop
