"""
Tests for SfM stages' _build_env (issue #10).

When the executable is a bare command name (to be resolved via PATH),
_build_env must NOT prepend the current working directory to PATH —
Path("colmap").resolve() is CWD-relative, so the old code silently put
the CWD at the front of PATH for every subprocess.
"""
from __future__ import annotations
import os
from pathlib import Path

from config import Config
from pipeline.stages.sfm.colmap import COLMAPStage
from pipeline.stages.sfm.colmap_panorama import PanoramaSFMStage
from pipeline.stages.sfm.spheresfm import SphereSFMStage


def _cfg_with_unset_tools() -> Config:
    cfg = Config()
    cfg.colmap_paths.colmappath = Path("")
    cfg.tool_paths.colmap_sphere = Path("")
    return cfg


# ── bare command name → no env override, CWD must not enter PATH ─────────

def test_colmap_bare_name_yields_no_env():
    assert COLMAPStage(_cfg_with_unset_tools())._env is None


def test_panorama_bare_name_yields_no_env():
    assert PanoramaSFMStage(_cfg_with_unset_tools(), mapper="global")._env is None


def test_spheresfm_bare_name_yields_no_env():
    assert SphereSFMStage(_cfg_with_unset_tools())._env is None


# ── explicit path → exe dir prepended exactly once, and it is not CWD ────

def _fake_exe(tmp_path: Path) -> Path:
    exe = tmp_path / "bin" / "colmap.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    return exe


def test_explicit_colmappath_prepends_exe_dir(tmp_path):
    exe = _fake_exe(tmp_path)
    cfg = _cfg_with_unset_tools()
    cfg.colmap_paths.colmappath = exe

    for stage in (COLMAPStage(cfg), PanoramaSFMStage(cfg, mapper="global")):
        env = stage._env
        assert env is not None
        first = env["PATH"].split(os.pathsep)[0]
        assert first == str(exe.parent.resolve())


def test_explicit_colmap_sphere_prepends_exe_dir(tmp_path):
    exe = _fake_exe(tmp_path)
    cfg = _cfg_with_unset_tools()
    cfg.tool_paths.colmap_sphere = exe

    env = SphereSFMStage(cfg)._env
    assert env is not None
    assert env["PATH"].split(os.pathsep)[0] == str(exe.parent.resolve())


def test_explicit_but_missing_dir_yields_no_env(tmp_path):
    cfg = _cfg_with_unset_tools()
    cfg.colmap_paths.colmappath = tmp_path / "nope" / "colmap.exe"
    assert COLMAPStage(cfg)._env is None
