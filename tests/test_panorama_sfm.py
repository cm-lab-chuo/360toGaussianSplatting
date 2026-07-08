"""
Unit tests for PanoramaSFMStage — fake `run` only, no COLMAP binary needed.

Verifies the orchestration contract from docs/360sfm_implementation_plan.md
Phase 2:
  * global route calls view_graph_calibrator + global_mapper
  * incremental route calls mapper
  * ctx.sparse_dir is set to work_dir/sparse
  * explicit failure when neither subcommand nor example script is available
"""
from __future__ import annotations
import logging
from pathlib import Path

import pytest

from config import Config
from pipeline.context import PipelineContext
from pipeline.stages.sfm import colmap_panorama
from pipeline.stages.sfm.colmap_panorama import PanoramaSFMStage
from registry import SFM

ALL_COMMANDS = {"panorama_sfm", "view_graph_calibrator", "global_mapper", "mapper"}


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def ctx(tmp_path: Path) -> PipelineContext:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_0001.jpg").write_bytes(b"fake")
    c = PipelineContext(input_path=tmp_path / "video.mp4", work_dir=tmp_path)
    c.frames_dir = frames
    return c


@pytest.fixture
def calls(monkeypatch) -> list[list[str]]:
    """Replace utils.process.run inside the stage module; record commands."""
    recorded: list[list[str]] = []

    def fake_run(cmd, cwd=None, env=None):
        recorded.append([str(c) for c in cmd])
        return 0

    monkeypatch.setattr(colmap_panorama, "run", fake_run)
    return recorded


def make_stage(
    monkeypatch,
    mapper: str,
    commands: set[str] = ALL_COMMANDS,
    **settings,
) -> PanoramaSFMStage:
    cfg = Config()
    for key, value in settings.items():
        setattr(cfg.panorama_sfm, key, value)
    stage = PanoramaSFMStage(cfg, mapper=mapper)
    monkeypatch.setattr(stage, "_colmap_commands", lambda: set(commands))
    monkeypatch.setattr(stage, "_colmap_help", lambda: "COLMAP 4.1")
    return stage


def subcommands(calls: list[list[str]]) -> list[str]:
    """The COLMAP subcommand (argv[1]) of each recorded call."""
    return [c[1] for c in calls]


# ── global mapper route ──────────────────────────────────────────────────

def test_global_calls_view_graph_calibrator_and_global_mapper(monkeypatch, ctx, calls):
    stage = make_stage(monkeypatch, "global")
    stage.run(ctx)
    assert subcommands(calls) == ["panorama_sfm", "view_graph_calibrator", "global_mapper"]


def test_global_without_view_graph_calibrator(monkeypatch, ctx, calls):
    stage = make_stage(monkeypatch, "global", run_view_graph_calibrator=False)
    stage.run(ctx)
    assert subcommands(calls) == ["panorama_sfm", "global_mapper"]


def test_global_requires_global_mapper_command(monkeypatch, ctx, calls):
    stage = make_stage(monkeypatch, "global", commands={"panorama_sfm", "mapper"})
    with pytest.raises(RuntimeError, match="global_mapper"):
        stage.run(ctx)


# ── incremental mapper route ─────────────────────────────────────────────

def test_incremental_calls_mapper(monkeypatch, ctx, calls):
    stage = make_stage(monkeypatch, "incremental")
    stage.run(ctx)
    assert subcommands(calls) == ["panorama_sfm", "mapper"]
    assert "view_graph_calibrator" not in subcommands(calls)
    assert "global_mapper" not in subcommands(calls)


# ── outputs / context ────────────────────────────────────────────────────

def test_sets_sparse_dir(monkeypatch, ctx, calls):
    stage = make_stage(monkeypatch, "global")
    result = stage.run(ctx)
    assert result.sparse_dir == ctx.work_dir / "sparse"
    assert result.sparse_dir.is_dir()


def test_prepare_uses_erp_frames_not_cubemap(monkeypatch, ctx, calls):
    cubemap = ctx.work_dir / "cubemap"
    cubemap.mkdir()
    (cubemap / "crop.jpg").write_bytes(b"fake")
    ctx.cubemap_dir = cubemap

    stage = make_stage(monkeypatch, "global")
    stage.run(ctx)
    prepare = calls[0]
    image_path = prepare[prepare.index("--image_path") + 1]
    assert image_path == str(ctx.frames_dir)


def test_requires_frames_dir(monkeypatch, ctx, calls):
    ctx.frames_dir = None
    stage = make_stage(monkeypatch, "global")
    with pytest.raises(RuntimeError, match="frames"):
        stage.run(ctx)


# ── keep_intermediate handling ───────────────────────────────────────────

def _make_views_dir(ctx) -> Path:
    views_dir = ctx.work_dir / "panorama" / "images"
    views_dir.mkdir(parents=True, exist_ok=True)
    (views_dir / "v_0001.jpg").write_bytes(b"fake")
    return views_dir


def test_keep_intermediate_false_warns_and_removes(monkeypatch, ctx, calls, caplog):
    stage = make_stage(monkeypatch, "global", keep_intermediate=False)
    views_dir = _make_views_dir(ctx)
    with caplog.at_level(logging.WARNING):
        stage.run(ctx)
    assert not views_dir.exists()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("keep_intermediate=False" in r.getMessage() for r in warnings)


def test_keep_intermediate_true_keeps_views_no_warning(monkeypatch, ctx, calls, caplog):
    stage = make_stage(monkeypatch, "global", keep_intermediate=True)
    views_dir = _make_views_dir(ctx)
    with caplog.at_level(logging.WARNING):
        stage.run(ctx)
    assert views_dir.exists()
    assert not any(
        "keep_intermediate" in r.getMessage()
        for r in caplog.records if r.levelno >= logging.WARNING
    )


# ── route detection / failure modes ──────────────────────────────────────

def test_no_route_raises_explicit_error(monkeypatch, ctx, calls):
    # COLMAP 3.8-like build: no panorama_sfm, no script configured.
    stage = make_stage(monkeypatch, "global", commands={"mapper"})
    with pytest.raises(RuntimeError, match="panorama_sfm is not available"):
        stage.run(ctx)
    assert calls == []


def test_script_route_used_as_fallback(monkeypatch, ctx, calls, tmp_path):
    script = tmp_path / "panorama_sfm.py"
    script.write_text("# example script", encoding="utf-8")
    stage = make_stage(
        monkeypatch, "incremental",
        commands={"mapper"},          # no panorama_sfm subcommand
        panorama_script=script,
    )
    stage.run(ctx)
    assert str(script) in calls[0]            # prepare via python script
    assert subcommands(calls)[-1] == "mapper"  # then incremental mapping


# ── mask limitation warning (Issue #8) ───────────────────────────────────

def _make_mask_dir(ctx) -> Path:
    mask_dir = ctx.work_dir / "masked" / "mask_only"
    mask_dir.mkdir(parents=True)
    (mask_dir / "frame_0001.mask.png").write_bytes(b"fake")
    return mask_dir


def test_warns_when_masker_output_would_be_ignored(monkeypatch, ctx, calls, caplog):
    ctx.mask_dir = _make_mask_dir(ctx)
    stage = make_stage(monkeypatch, "global")
    with caplog.at_level("WARNING"):
        stage.run(ctx)
    assert any("WITHOUT masks" in r.message for r in caplog.records)


def test_warns_when_external_masks_would_be_ignored(monkeypatch, ctx, calls, caplog, tmp_path):
    ext = tmp_path / "external_masks"
    ext.mkdir()
    (ext / "frame_0001.png").write_bytes(b"fake")
    stage = make_stage(monkeypatch, "global")
    stage.cfg.mask.external_mask_dir = ext
    with caplog.at_level("WARNING"):
        stage.run(ctx)
    assert any("WITHOUT masks" in r.message for r in caplog.records)


def test_no_mask_warning_when_masking_disabled(monkeypatch, ctx, calls, caplog):
    ctx.mask_dir = _make_mask_dir(ctx)
    stage = make_stage(monkeypatch, "global")
    stage.cfg.mask.apply_masks_to_sfm = False
    with caplog.at_level("WARNING"):
        stage.run(ctx)
    assert not any("WITHOUT masks" in r.message for r in caplog.records)


def test_no_mask_warning_when_no_masks_exist(monkeypatch, ctx, calls, caplog):
    stage = make_stage(monkeypatch, "global")
    with caplog.at_level("WARNING"):
        stage.run(ctx)
    assert not any("WITHOUT masks" in r.message for r in caplog.records)


# ── registry / config integration ────────────────────────────────────────

def test_registry_entries_set_mapper():
    cfg = Config()
    assert SFM["panorama_global"](cfg).mapper == "global"
    assert SFM["panorama_incremental"](cfg).mapper == "incremental"


def test_invalid_mapper_rejected():
    with pytest.raises(ValueError, match="mapper"):
        PanoramaSFMStage(Config(), mapper="hierarchical")
