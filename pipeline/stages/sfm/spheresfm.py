"""
Stage 5a — SphereSFM (COLMAP variant for 360° / spherical images).

Uses a colmap_sphere.exe binary. This variant accounts for the geometric
relationship between perspective crops taken from the same equirectangular
frame (realign_cubemaps=True), enabling stronger pose constraints than
standard COLMAP.

SfM steps (sequential):
  1. feature_extractor  — SIFT on each perspective crop
  2. sequential_matcher — match frame N ↔ N±k (exploits temporal order)
  3. mapper             — incremental SfM
  4. (optional) bundle_adjuster — global refinement

Output (COLMAP sparse format, ready for 3DGS trainers):
  ctx.sparse_dir/
    0/
      cameras.bin
      images.bin
      points3D.bin

Configurable parameters: SphereSFMSettings (all forwarded to colmap_sphere).
The executable path is read from [ToolPaths] colmap_sphere (or resolved from
PATH); it can also be passed explicitly via exe_path.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext
from utils.process import run
from utils.masks import resolve_colmap_mask_path

logger = logging.getLogger(__name__)


class SphereSFMStage(Stage):

    def __init__(self, cfg: Config, exe_path: Path | None = None) -> None:
        super().__init__(cfg)
        configured = cfg.tool_paths.colmap_sphere
        is_set = str(configured) not in ("", ".")
        self._exe = exe_path or (configured if is_set else Path("colmap_sphere.exe"))
        self._env = self._build_env()

    def _build_env(self) -> dict | None:
        """Add the exe's directory to PATH so Windows can find sibling DLLs."""
        exe_dir = Path(self._exe).resolve().parent
        if not exe_dir.is_dir():
            return None
        env = os.environ.copy()
        env["PATH"] = str(exe_dir) + os.pathsep + env.get("PATH", "")
        return env

    @property
    def name(self) -> str:
        return "SphereSFM"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not self._exe.exists():
            raise FileNotFoundError(
                f"colmap_sphere.exe not found at {self._exe}\n"
                "Set [ToolPaths] colmap_sphere in your config, add it to PATH, "
                "or pass exe_path= to SphereSFMStage."
            )

        images_dir = ctx.images_for_sfm()
        assert images_dir is not None and images_dir.exists()

        sparse_dir = ctx.stage_dir("sparse")
        db_path = ctx.work_dir / "database.db"
        s = self.cfg.spheresfm

        # Resolve masks from any source (masker stage or external program) into
        # COLMAP convention; None if masking is off / unavailable.
        mask_path = resolve_colmap_mask_path(ctx, self.cfg)

        # 1 — feature extraction
        self._run_feature_extractor(images_dir, db_path, s, mask_path)

        # 2 — feature matching
        self._run_matcher(db_path, s)

        # 3 — mapping
        model_dir = sparse_dir / "0"
        model_dir.mkdir(parents=True, exist_ok=True)
        self._run_mapper(images_dir, db_path, sparse_dir, s)

        # 4 — optional bundle adjustment
        if s.run_bundle_adjuster:
            self._run_bundle_adjuster(model_dir, s)

        ctx.sparse_dir = sparse_dir
        logger.info("SphereSFM complete → %s", sparse_dir)
        return ctx

    # ── sub-commands ─────────────────────────────────────────────────────

    def _run_feature_extractor(self, images_dir: Path, db: Path, s,
                               mask_path: Path | None = None) -> None:
        cmd = [
            self._exe, "feature_extractor",
            "--database_path", db,
            "--image_path", images_dir,
            "--SiftExtraction.max_num_features", s.max_num_features,
            "--SiftExtraction.first_octave", s.first_octave,
            "--SiftExtraction.peak_threshold", s.peak_threshold,
            "--SiftExtraction.edge_threshold", s.edge_threshold,
            "--SiftExtraction.estimate_affine_shape",
                "1" if s.estimate_affine_shape else "0",
            "--SiftExtraction.domain_size_pooling",
                "1" if s.domain_size_pooling else "0",
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
        ]
        if mask_path is not None:
            cmd += ["--ImageReader.mask_path", mask_path]
        if s.realign_cubemaps:
            cmd += ["--realign_cubemaps", "1"]
        run(cmd, env=self._env)

    def _run_matcher(self, db: Path, s) -> None:
        matcher_cmd = (
            "sequential_matcher"
            if s.matcher_type == "sequential"
            else "exhaustive_matcher"
        )
        cmd = [
            self._exe, matcher_cmd,
            "--database_path", db,
            "--SiftMatching.max_num_matches", s.max_num_matches,
            "--SiftMatching.guided_matching", "1" if s.guided_matching else "0",
        ]
        run(cmd, env=self._env)

    def _run_mapper(self, images_dir: Path, db: Path, sparse_dir: Path, s) -> None:
        cmd = [
            self._exe, "mapper",
            "--database_path", db,
            "--image_path", images_dir,
            "--output_path", sparse_dir,
            "--Mapper.ba_local_max_num_iterations", s.ba_local_max_iterations,
            "--Mapper.ba_global_max_num_iterations", s.ba_global_max_iterations,
            "--Mapper.ba_global_max_refinements", s.ba_global_max_refinements,
            "--Mapper.filter_max_reproj_error", s.filter_max_reproj_error,
            "--Mapper.filter_min_tri_angle", s.filter_min_tri_angle,
            "--Mapper.abs_pose_min_num_inliers", s.abs_pose_min_num_inliers,
        ]
        if s.cubemap_refine_focal:
            cmd += ["--cubemap_refine_focal", "1"]
        run(cmd, env=self._env)

    def _run_bundle_adjuster(self, model_dir: Path, s) -> None:
        cmd = [
            self._exe, "bundle_adjuster",
            "--input_path", model_dir,
            "--output_path", model_dir,
            "--BundleAdjustment.max_num_iterations", s.ba_global_max_iterations,
        ]
        run(cmd, env=self._env)
