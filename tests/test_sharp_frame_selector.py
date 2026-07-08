"""Unit tests for sharp frame extraction (Issue #15)."""
from __future__ import annotations

import numpy as np
import pytest

from config import Config, VideoSettings
from pipeline.stages.preprocessing import sharp_frame_selector as sfs


# ── interval_length ───────────────────────────────────────────────────────


def test_interval_length_seconds_mode():
    # 2 frames/sec out of a 30 fps source → one interval = 15 source frames
    assert sfs.interval_length("seconds", 2.0, 30.0) == 15.0


def test_interval_length_frames_mode_uses_fps_as_step():
    assert sfs.interval_length("frames", 5.0, 30.0) == 5.0


def test_interval_length_clamps_to_one_frame():
    # Requesting more frames than the source has per second → keep every frame
    assert sfs.interval_length("seconds", 120.0, 30.0) == 1.0


def test_interval_length_rejects_non_positive_fps():
    with pytest.raises(ValueError):
        sfs.interval_length("seconds", 0.0, 30.0)


def test_interval_length_rejects_unknown_mode():
    with pytest.raises(ValueError):
        sfs.interval_length("minutes", 1.0, 30.0)


# ── interval_bounds ───────────────────────────────────────────────────────


def test_interval_bounds_are_contiguous_and_non_empty():
    for interval in (1.0, 2.5, 7.5, 15.0, 29.97):
        prev_end = 0
        for k in range(50):
            start, end = sfs.interval_bounds(k, interval)
            assert start == prev_end, f"gap/overlap at k={k}, interval={interval}"
            assert end > start
            prev_end = end


# ── candidate_window ──────────────────────────────────────────────────────


def test_window_zero_range_selects_whole_interval():
    assert sfs.candidate_window(30, 60, 0) == (30, 60)


def test_window_larger_than_interval_selects_whole_interval():
    assert sfs.candidate_window(30, 40, 10) == (30, 40)
    assert sfs.candidate_window(30, 40, 25) == (30, 40)


def test_window_is_centered_in_interval():
    # interval [0, 30), center 15, 10-frame window → [10, 20)
    assert sfs.candidate_window(0, 30, 10) == (10, 20)


def test_window_has_requested_size_when_it_fits():
    for check_range in (1, 3, 7, 10):
        lo, hi = sfs.candidate_window(100, 130, check_range)
        assert hi - lo == check_range
        assert 100 <= lo < hi <= 130


def test_window_clamps_to_interval_bounds():
    lo, hi = sfs.candidate_window(0, 10, 8)
    assert (lo, hi) == (1, 9)
    lo, hi = sfs.candidate_window(0, 10, 9)
    assert 0 <= lo < hi <= 10 and hi - lo == 9


# ── select_sharpest ───────────────────────────────────────────────────────


def test_select_sharpest_picks_max_score():
    assert sfs.select_sharpest([(3, 1.0), (4, 9.5), (5, 2.0)]) == 4


def test_select_sharpest_tie_goes_to_earliest():
    assert sfs.select_sharpest([(7, 5.0), (8, 5.0), (9, 5.0)]) == 7


def test_select_sharpest_empty_raises():
    with pytest.raises(ValueError):
        sfs.select_sharpest([])


# ── fit_and_pad ───────────────────────────────────────────────────────────


def test_fit_and_pad_output_shape_and_centering():
    img = np.full((100, 200, 3), 255, dtype=np.uint8)  # wide white image
    out = sfs.fit_and_pad(img, 64, 64)
    assert out.shape == (64, 64, 3)
    # 200x100 fits 64x64 as 64x32 → 16px black bands top and bottom
    assert out[0, 32].tolist() == [0, 0, 0]
    assert out[63, 32].tolist() == [0, 0, 0]
    assert out[32, 32].tolist() == [255, 255, 255]


def test_fit_and_pad_no_padding_when_aspect_matches():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    out = sfs.fit_and_pad(img, 64, 64)
    assert out.shape == (64, 64, 3)
    assert out.min() == 200  # no black border anywhere


# ── config plumbing ───────────────────────────────────────────────────────


def test_video_settings_defaults():
    v = VideoSettings()
    assert v.sharp_frame_extraction is False
    assert v.sharpness_check_range == 10


def test_from_ini_parses_new_keys(tmp_path):
    ini = tmp_path / "exp.ini"
    ini.write_text(
        "[VideoSettings]\n"
        "sharp_frame_extraction = True\n"
        "sharpness_check_range = 21\n",
        encoding="utf-8",
    )
    cfg = Config.from_ini(ini)
    assert cfg.video.sharp_frame_extraction is True
    assert cfg.video.sharpness_check_range == 21


# ── extraction loop (fake VideoCapture, no video files needed) ────────────


class _FakeCapture:
    """Stands in for cv2.VideoCapture: emits pre-built frames at 30 fps."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._pos = 0

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        if self._pos >= len(self._frames):
            return False, None
        f = self._frames[self._pos]
        self._pos += 1
        return True, f

    def release(self):
        pass


def _make_frames(n, sharp_indices, size=64):
    """Flat gray frames (value 100+i, near-zero variance); the designated
    indices get a checkerboard overlay → high Laplacian variance."""
    frames = []
    checker = np.indices((size, size)).sum(axis=0) % 2 * 255
    for i in range(n):
        f = np.full((size, size, 3), 100 + i, dtype=np.uint8)
        if i in sharp_indices:
            f[:, :, 0] = checker.astype(np.uint8)
        frames.append(f)
    return frames


def _settings(**kw):
    v = VideoSettings()
    v.sharp_frame_extraction = True
    v.frame_extraction_mode = "seconds"
    v.fps = 1.0                      # @30fps fake source → 30-frame intervals
    v.resolutionwidth = 64
    v.resolutionheight = 64
    for k, val in kw.items():
        setattr(v, k, val)
    return v


def test_extract_picks_sharp_frame_per_interval(tmp_path, monkeypatch):
    import cv2
    # 90 frames = 3 intervals; sharp frames at 5, 40, 70 (check whole interval)
    frames = _make_frames(90, sharp_indices={5, 40, 70})
    monkeypatch.setattr(sfs.cv2, "VideoCapture", lambda _: _FakeCapture(frames))

    out = tmp_path / "frames"
    out.mkdir()
    n = sfs.extract_sharpest_frames(tmp_path / "fake.mp4", out, _settings(sharpness_check_range=0))

    assert n == 3
    written = sorted(out.glob("*.jpg"))
    assert [p.name for p in written] == ["000001.jpg", "000002.jpg", "000003.jpg"]
    # Each output should be the checkerboard frame, not a flat one
    for p in written:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        # JPEG smooths the 1-px checkerboard, but a flat frame stays near 0.
        assert float(img.std()) > 10.0, f"{p.name} is not the sharp frame"


def test_extract_respects_check_range_window(tmp_path, monkeypatch):
    import cv2
    # Sharp frame at index 2 sits OUTSIDE the 10-frame window centered at 15
    # ([10, 20)), so the flat center frame must win instead.
    frames = _make_frames(30, sharp_indices={2})
    monkeypatch.setattr(sfs.cv2, "VideoCapture", lambda _: _FakeCapture(frames))

    out = tmp_path / "frames"
    out.mkdir()
    n = sfs.extract_sharpest_frames(tmp_path / "fake.mp4", out, _settings(sharpness_check_range=10))

    assert n == 1
    img = cv2.imdecode(
        np.fromfile(str(out / "000001.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert float(img.std()) < 5.0, "picked the out-of-window checkerboard frame"


def test_extract_handles_partial_last_interval(tmp_path, monkeypatch):
    # 75 frames @30fps, 1fps → intervals [0,30), [30,60), partial [60,75)
    frames = _make_frames(75, sharp_indices=set())
    monkeypatch.setattr(sfs.cv2, "VideoCapture", lambda _: _FakeCapture(frames))

    out = tmp_path / "frames"
    out.mkdir()
    n = sfs.extract_sharpest_frames(tmp_path / "fake.mp4", out, _settings(sharpness_check_range=0))
    assert n == 3  # the partial tail still yields one frame
