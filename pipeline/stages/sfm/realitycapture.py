"""
Stage 5c — RealityCapture / RealityScan alignment.

Calls RealityScan.exe (Epic Games) for camera registration.
After alignment, exports camera parameters in COLMAP-compatible format
so downstream 3DGS trainers can consume the result directly.

RealityCapture CLI reference:
  https://support.capturingreality.com/hc/en-us/articles/360017527431

Workflow:
  1. Create RC project from images
  2. Align cameras
  3. Export cameras as COLMAP format (via RC_Settings XMLs)
  4. Convert RC output → COLMAP sparse (cameras.bin, images.bin, points3D.bin)

Note: Step 3 uses RC export presets from the configured RC_Settings folder
([PostShotPaths] settingsfolder): 3DGS_reg.xml (camera params) and
3DGS_ply.xml (sparse point cloud as PLY).
"""
from __future__ import annotations
import logging
from pathlib import Path

from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext
from utils.process import run

logger = logging.getLogger(__name__)


class RealityCaptureStage(Stage):

    @property
    def name(self) -> str:
        return "RealityCapture"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        rc_exe = self.cfg.postshot_paths.realitycapturepath
        settings_folder = self.cfg.postshot_paths.settingsfolder

        if not rc_exe.exists():
            raise FileNotFoundError(
                f"RealityScan not found at {rc_exe}\n"
                "Update [PostShotPaths] realitycapturepath in your config."
            )

        images_dir = ctx.images_for_sfm()
        assert images_dir is not None

        sparse_dir = ctx.stage_dir("sparse")
        rc_project = ctx.work_dir / "rc_project.rcproj"
        export_dir = ctx.work_dir / "rc_export"
        export_dir.mkdir(exist_ok=True)

        reg_xml = settings_folder / "3DGS_reg.xml" if settings_folder.exists() else None
        ply_xml = settings_folder / "3DGS_ply.xml" if settings_folder.exists() else None

        # Build the RealityCapture command chain
        cmd = [rc_exe]

        # Add images
        cmd += ["-addFolder", images_dir]

        # Align cameras
        cmd += ["-align"]

        # Export camera registration
        if reg_xml and reg_xml.exists():
            cmd += ["-exportRegistration", export_dir / "cameras", reg_xml]

        # Export sparse point cloud as PLY
        if ply_xml and ply_xml.exists():
            cmd += ["-exportModel", "sparse_cloud", export_dir / "sparse.ply", ply_xml]

        # Save and quit
        cmd += ["-save", rc_project, "-quit"]

        run(cmd)

        # The export gives us RC-format camera params; convert to COLMAP format
        self._convert_to_colmap(export_dir, sparse_dir, images_dir)

        ctx.sparse_dir = sparse_dir
        logger.info("RealityCapture alignment complete → %s", sparse_dir)
        return ctx

    def _convert_to_colmap(
        self, export_dir: Path, sparse_dir: Path, images_dir: Path
    ) -> None:
        """
        Convert RealityCapture export → COLMAP sparse format.

        RealityCapture exports in its own XML/CSV format.  This conversion
        must be implemented once RC export format is confirmed.

        For now, the raw RC export is preserved so you can inspect it and
        implement the conversion appropriate for your RC version.
        """
        sparse_dir.mkdir(parents=True, exist_ok=True)
        model_dir = sparse_dir / "0"
        model_dir.mkdir(exist_ok=True)

        logger.warning(
            "RealityCapture → COLMAP conversion not yet implemented.\n"
            "Raw RC export is at: %s\n"
            "Implement _convert_to_colmap() or use an external converter "
            "(e.g. rc2colmap, or pycolmap to write cameras.bin/images.bin).",
            export_dir,
        )
