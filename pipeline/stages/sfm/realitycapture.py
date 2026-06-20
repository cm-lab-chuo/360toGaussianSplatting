"""RealityCapture / RealityScan camera alignment stage."""
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
        sparse_dir = ctx.stage_dir("sparse")
        rc_project = ctx.work_dir / "rc_project.rcproj"
        export_dir = ctx.work_dir / "rc_export"
        export_dir.mkdir(exist_ok=True)

        reg_xml = settings_folder / "3DGS_reg.xml" if settings_folder.exists() else None
        ply_xml = settings_folder / "3DGS_ply.xml" if settings_folder.exists() else None

        cmd = [rc_exe, "-addFolder", images_dir, "-align"]
        if reg_xml and reg_xml.exists():
            cmd += ["-exportRegistration", export_dir / "cameras", reg_xml]
        if ply_xml and ply_xml.exists():
            cmd += ["-exportModel", "sparse_cloud", export_dir / "sparse.ply", ply_xml]
        cmd += ["-save", rc_project, "-quit"]

        run(cmd)

        self._convert_to_colmap(export_dir, sparse_dir, images_dir)
        self._validate_colmap_model(sparse_dir / "0")

        ctx.sparse_dir = sparse_dir
        logger.info("RealityCapture alignment complete -> %s", sparse_dir)
        return ctx

    def _convert_to_colmap(
        self, export_dir: Path, sparse_dir: Path, images_dir: Path
    ) -> None:
        """Convert the RealityCapture export to a COLMAP sparse model."""
        raise NotImplementedError(
            "RealityCapture to COLMAP conversion is not implemented. "
            f"Raw RealityCapture output is preserved at {export_dir}. "
            "Implement _convert_to_colmap() before using --sfm realitycapture."
        )

    @staticmethod
    def _validate_colmap_model(model_dir: Path) -> None:
        required = ("cameras.bin", "images.bin", "points3D.bin")
        missing = [name for name in required if not (model_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Incomplete COLMAP model in {model_dir}: missing {missing}"
            )
