"""
Stage 5b — Standard COLMAP SfM.

Uses vanilla COLMAP for camera pose estimation.  Best suited when you have
non-360° images or want to compare against SphereSFM's 360°-aware variant.

Requires COLMAP to be on PATH, or [COLMAPPaths] colmappath set in config.

Output: same COLMAP sparse format as SphereSFM (ctx.sparse_dir/0/).
"""
from __future__ import annotations
import logging
from pathlib import Path

from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext
from utils.process import run
from utils.masks import resolve_colmap_mask_path

logger = logging.getLogger(__name__)


class COLMAPStage(Stage):

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        # An unset config value normalizes to Path("") == Path("."); treat "" and
        # "." as "not configured" and fall back to PATH resolution.
        colmap_path = cfg.colmap_paths.colmappath
        is_set = str(colmap_path) not in ("", ".")
        self._exe = colmap_path if is_set else Path("colmap")

    @property
    def name(self) -> str:
        return "COLMAP"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        images_dir = ctx.images_for_sfm()
        assert images_dir is not None and images_dir.exists()

        sparse_dir = ctx.stage_dir("sparse")
        db_path = ctx.work_dir / "database.db"
        s = self.cfg.spheresfm  # reuse same params (compatible subset)

        # Resolve masks from any source (masker stage or external program).
        mask_path = resolve_colmap_mask_path(ctx, self.cfg)

        # Feature extraction
        feat_cmd = [
            self._exe, "feature_extractor",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--SiftExtraction.max_num_features", s.max_num_features,
            "--SiftExtraction.first_octave", s.first_octave,
            "--SiftExtraction.peak_threshold", s.peak_threshold,
            "--SiftExtraction.edge_threshold", s.edge_threshold,
            "--SiftExtraction.estimate_affine_shape",
                "1" if s.estimate_affine_shape else "0",
            "--SiftExtraction.domain_size_pooling",
                "1" if s.domain_size_pooling else "0",
            "--ImageReader.camera_model", "PINHOLE",
        ]
        if mask_path is not None:
            feat_cmd += ["--ImageReader.mask_path", mask_path]
        run(feat_cmd)

        # Matching
        matcher = "sequential_matcher" if s.matcher_type == "sequential" else "exhaustive_matcher"
        run([
            self._exe, matcher,
            "--database_path", db_path,
            "--SiftMatching.max_num_matches", s.max_num_matches,
            "--SiftMatching.guided_matching", "1" if s.guided_matching else "0",
        ])

        # Mapping
        model_dir = sparse_dir / "0"
        model_dir.mkdir(parents=True, exist_ok=True)
        run([
            self._exe, "mapper",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--output_path", sparse_dir,
            "--Mapper.ba_local_max_num_iterations", s.ba_local_max_iterations,
            "--Mapper.ba_global_max_num_iterations", s.ba_global_max_iterations,
            "--Mapper.filter_max_reproj_error", s.filter_max_reproj_error,
            "--Mapper.filter_min_tri_angle", s.filter_min_tri_angle,
            "--Mapper.abs_pose_min_num_inliers", s.abs_pose_min_num_inliers,
        ])

        ctx.sparse_dir = sparse_dir
        logger.info("COLMAP complete → %s", sparse_dir)
        return ctx
