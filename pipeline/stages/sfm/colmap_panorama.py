"""
Stage 5c — COLMAP 4.1 panorama_sfm (ERP frames + virtual rig).

First-choice method from docs/360sfm_implementation_plan.md (Phase 2):
render virtual perspective views from the RAW equirectangular frames so the
"views from the same 360° frame form a rig" relation is preserved, then
reconstruct with either the global or the incremental mapper.

IMPORTANT: input is ctx.frames_dir (ERP), NOT the cubemap/ crops — feeding
pre-split crops would drop the rig relation the method depends on.

LIMITATION (Issue #8): masks are NOT supported yet. panorama_sfm renders its
own virtual views and runs feature extraction internally, so there is no hook
to pass --ImageReader.mask_path (masks would have to be reprojected to the
virtual views with the same parameters first). If masks are available they are
ignored, and run() emits an explicit WARNING so the omission is visible.

Route selection (checked at run time, in order):
  1. Local COLMAP has a `panorama_sfm` subcommand (COLMAP >= 4.1) → call it.
  2. [PanoramaSFMSettings] panorama_script points to the pycolmap example
     script (colmap/python/examples/panorama_sfm.py) → run it via Python.
  3. Neither → fail with an explicit message (docs/colmap_41_environment.md).

Steps within this stage:
  prepare — panorama_sfm renders virtual views + builds the COLMAP database
            (rig constraints, features, matches); mapping is skipped
  map     — global:      [view_graph_calibrator →] global_mapper
            incremental: mapper

Output: COLMAP sparse model in ctx.work_dir/"sparse" (same layout as the
other SfM stages); ctx.sparse_dir is set accordingly.

NOTE: the exact flag names of `panorama_sfm` must be verified against the
local COLMAP 4.1 distribution (Phase 0). The flags below encode the
interface assumed in the implementation plan; adjust here (or via
panorama_script_extra_args for the script route) once fixed.
"""
from __future__ import annotations
import logging
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext
from utils.process import run, run_capture

logger = logging.getLogger(__name__)

# Minimum COLMAP version expected to ship panorama_sfm / global_mapper.
REQUIRED_COLMAP_VERSION = (4, 1)


class PanoramaSFMStage(Stage):

    def __init__(self, cfg: Config, mapper: str | None = None) -> None:
        super().__init__(cfg)
        self.mapper = (mapper or cfg.panorama_sfm.mapper).lower()
        if self.mapper not in ("global", "incremental"):
            raise ValueError(
                f"PanoramaSFMSettings.mapper must be 'global' or 'incremental', "
                f"got {self.mapper!r}"
            )
        colmap_path = cfg.colmap_paths.colmappath
        is_set = str(colmap_path) not in ("", ".")
        self._exe = colmap_path if is_set else Path("colmap")
        self._env = self._build_env()
        self._help_text: str | None = None  # cached `colmap help` output

    def _build_env(self) -> dict | None:
        """Prepend the exe's directory to PATH so sibling DLLs are found.

        Skipped for bare command names (exe to be resolved via PATH):
        resolving a bare name is relative to the CWD, which would wrongly
        prepend the current working directory to PATH (issue #10).
        """
        exe = Path(self._exe)
        if exe.parent == Path("."):  # bare name — no directory component
            return None
        exe_dir = exe.resolve().parent
        if not exe_dir.is_dir():
            return None
        env = os.environ.copy()
        env["PATH"] = str(exe_dir) + os.pathsep + env.get("PATH", "")
        return env

    @property
    def name(self) -> str:
        return f"PanoramaSFM({self.mapper})"

    # ── COLMAP capability probing ────────────────────────────────────────

    def _colmap_help(self) -> str:
        if self._help_text is None:
            _, out = run_capture([self._exe, "help"], env=self._env)
            self._help_text = out
        return self._help_text

    def _colmap_version(self) -> tuple[int, int] | None:
        m = re.search(r"COLMAP\s+(\d+)\.(\d+)", self._colmap_help())
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _colmap_commands(self) -> set[str]:
        """Subcommands listed under 'Available commands:' in `colmap help`."""
        cmds: set[str] = set()
        in_list = False
        for line in self._colmap_help().splitlines():
            if line.strip() == "Available commands:":
                in_list = True
                continue
            if in_list:
                token = line.strip()
                if re.fullmatch(r"[a-z0-9_]+", token):
                    cmds.add(token)
        return cmds

    def _detect_route(self) -> str:
        """Return 'subcommand' or 'script'; raise if neither is available."""
        if "panorama_sfm" in self._colmap_commands():
            return "subcommand"
        script = self.cfg.panorama_sfm.panorama_script
        if str(script) not in ("", ".") and Path(script).is_file():
            return "script"
        version = self._colmap_version()
        version_str = ".".join(map(str, version)) if version else "not found"
        raise RuntimeError(
            f"panorama_sfm is not available.\n"
            f"  Local COLMAP: {self._exe} (version: {version_str}, "
            f"required: >= {'.'.join(map(str, REQUIRED_COLMAP_VERSION))})\n"
            f"  - No `panorama_sfm` subcommand in this COLMAP build, and\n"
            f"  - [PanoramaSFMSettings] panorama_script is not set / not a file.\n"
            f"Fix: install COLMAP 4.1+, or set panorama_script to the pycolmap "
            f"example (colmap/python/examples/panorama_sfm.py). "
            f"See docs/colmap_41_environment.md."
        )

    # ── stage entry point ────────────────────────────────────────────────

    def run(self, ctx: PipelineContext) -> PipelineContext:
        s = self.cfg.panorama_sfm
        self._warn_if_masks_ignored(ctx)
        images_dir = self._resolve_input(ctx)
        sparse_dir = ctx.stage_dir("sparse")
        pano_dir = ctx.work_dir / "panorama"
        pano_dir.mkdir(parents=True, exist_ok=True)

        route = self._detect_route()
        version = self._colmap_version()
        logger.info(
            "PanoramaSFM route=%s mapper=%s (COLMAP %s)",
            route, self.mapper,
            ".".join(map(str, version)) if version else "unknown",
        )

        if route == "subcommand":
            db_path, views_dir = self._prepare_subcommand(images_dir, pano_dir)
        else:
            db_path, views_dir = self._prepare_script(images_dir, pano_dir)

        self._run_mapping(db_path, views_dir, sparse_dir)

        if not s.keep_intermediate:
            logger.warning(
                "keep_intermediate=False: deleting the virtual perspective views "
                "(%s). The sparse model's image entries point to these files, so "
                "downstream steps that read the images (3DGS training, "
                "`colmap image_undistorter`, dense reconstruction, ...) will NOT "
                "find them and will fail or silently drop frames. Keep "
                "[PanoramaSFMSettings] keep_intermediate = True if anything "
                "after SfM consumes the sparse model.",
                views_dir,
            )
            shutil.rmtree(views_dir, ignore_errors=True)
            logger.info("Removed intermediate virtual views: %s", views_dir)

        ctx.sparse_dir = sparse_dir
        logger.info("PanoramaSFM (%s) complete → %s", self.mapper, sparse_dir)
        return ctx

    def _warn_if_masks_ignored(self, ctx: PipelineContext) -> None:
        """
        panorama_sfm has no hook for --ImageReader.mask_path (Issue #8): it
        renders virtual views and extracts features internally. If masks are
        available and would normally be applied to SfM, say so loudly instead
        of silently reconstructing without them.
        """
        ms = self.cfg.mask
        if not ms.apply_masks_to_sfm:
            return

        def _usable(d) -> bool:
            return (bool(d) and str(d) not in ("", ".")
                    and Path(d).exists() and any(Path(d).iterdir()))

        source = None
        if _usable(ms.external_mask_dir):
            source = f"[MaskSettings] external_mask_dir = {ms.external_mask_dir}"
        elif _usable(ctx.mask_dir):
            source = f"masker stage output ({ctx.mask_dir})"
        if source:
            logger.warning(
                "Masks are available (%s) but panorama_sfm does NOT support "
                "masking yet — reconstruction will run WITHOUT masks. Dynamic "
                "objects may leak into the model. Use --sfm spheresfm/colmap "
                "if masked SfM is required, or skip the masker "
                "(--masker none --skip cubemap) to save compute.",
                source,
            )

    def _resolve_input(self, ctx: PipelineContext) -> Path:
        d = ctx.frames_dir
        if d and Path(d).is_dir() and any(Path(d).iterdir()):
            return Path(d)
        raise RuntimeError(
            "panorama_sfm requires the RAW equirectangular frames (ctx.frames_dir), "
            "but frames/ is empty or unset. Run the extraction stage first "
            "(or resume with --skip extraction so frames/ is restored). "
            "Cubemap crops are NOT a valid input — the rig relation would be lost."
        )

    # ── prepare: render virtual views + build database ───────────────────

    def _prepare_subcommand(self, images_dir: Path, pano_dir: Path) -> tuple[Path, Path]:
        s = self.cfg.panorama_sfm
        db_path = pano_dir / "database.db"
        views_dir = pano_dir / "images"
        run([
            self._exe, "panorama_sfm",
            "--image_path", images_dir,
            "--database_path", db_path,
            "--output_path", views_dir,
            "--num_virtual_views", s.num_virtual_views,
            "--virtual_view_fov", s.virtual_view_fov,
            "--camera_model", s.camera_model,
            "--use_gpu", "1" if s.use_gpu else "0",
            "--skip_mapping", "1",
        ], env=self._env)
        return db_path, views_dir

    def _prepare_script(self, images_dir: Path, pano_dir: Path) -> tuple[Path, Path]:
        s = self.cfg.panorama_sfm
        python = (
            str(s.python_exe)
            if str(s.python_exe) not in ("", ".")
            else sys.executable
        )
        cmd = [
            python, s.panorama_script,
            "--input_image_path", images_dir,
            "--output_path", pano_dir,
            "--num_virtual_views", s.num_virtual_views,
            "--virtual_view_fov", s.virtual_view_fov,
            "--camera_model", s.camera_model,
            "--skip_mapping",
        ]
        extra = s.panorama_script_extra_args.strip()
        if extra:
            cmd += shlex.split(extra)
        run(cmd, env=self._env)
        # The example script writes its database and rendered views under
        # output_path — adjust here if the local script differs (Phase 0).
        return pano_dir / "database.db", pano_dir / "images"

    # ── map: global or incremental ───────────────────────────────────────

    def _run_mapping(self, db_path: Path, views_dir: Path, sparse_dir: Path) -> None:
        s = self.cfg.panorama_sfm
        (sparse_dir / "0").mkdir(parents=True, exist_ok=True)

        if self.mapper == "global":
            needed = {"global_mapper"}
            if s.run_view_graph_calibrator:
                needed.add("view_graph_calibrator")
            missing = needed - self._colmap_commands()
            if missing:
                raise RuntimeError(
                    f"Global mapping needs COLMAP subcommand(s) "
                    f"{', '.join(sorted(missing))} which this build lacks "
                    f"({self._exe}). Install COLMAP 4.1+, or use "
                    f"--sfm panorama_incremental instead."
                )
            if s.run_view_graph_calibrator:
                run([
                    self._exe, "view_graph_calibrator",
                    "--database_path", db_path,
                ], env=self._env)
            run([
                self._exe, "global_mapper",
                "--database_path", db_path,
                "--image_path", views_dir,
                "--output_path", sparse_dir,
            ], env=self._env)
        else:
            run([
                self._exe, "mapper",
                "--database_path", db_path,
                "--image_path", views_dir,
                "--output_path", sparse_dir,
            ], env=self._env)
