"""
Sharp frame extraction — pick the sharpest frame per extraction interval.

Instead of mechanically taking the first frame at each rate tick (the ffmpeg
path), this module walks the video sequentially with OpenCV, evaluates a
window of candidate frames around each interval's center, and keeps only the
single sharpest frame (Laplacian variance, same metric as FrameFilter).

Configuration (VideoSettings):
  sharp_frame_extraction — enables this path (video input only)
  sharpness_check_range  — window size in frames, centered in the interval;
                           0 (or >= interval length) evaluates the whole interval

Pure helpers (interval_length / interval_bounds / candidate_window /
select_sharpest / fit_and_pad) are kept free of I/O so they can be unit-tested
without video fixtures.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

JPEG_QUALITY = 96  # ~ ffmpeg -q:v 2 (near-lossless)


# ── pure helpers ──────────────────────────────────────────────────────────


def interval_length(mode: str, fps: float, native_fps: float) -> float:
    """Length of one extraction interval, in source frames (>= 1.0)."""
    if fps <= 0:
        raise ValueError("VideoSettings.fps must be greater than zero")
    if mode == "seconds":
        iv = native_fps / fps
    elif mode == "frames":
        iv = float(max(1, round(fps)))
    else:
        raise ValueError(
            "VideoSettings.frame_extraction_mode must be 'seconds' or 'frames'"
        )
    return max(1.0, iv)


def interval_bounds(k: int, interval: float) -> tuple[int, int]:
    """Half-open frame-index range [start, end) of interval k."""
    start = round(k * interval)
    end = max(start + 1, round((k + 1) * interval))
    return start, end


def candidate_window(start: int, end: int, check_range: int) -> tuple[int, int]:
    """
    Half-open candidate range within [start, end).

    check_range <= 0 or >= interval length selects the whole interval;
    otherwise a `check_range`-frame window centered in the interval,
    clamped to the interval bounds.
    """
    length = end - start
    if check_range <= 0 or check_range >= length:
        return start, end
    center = start + length // 2
    lo = center - check_range // 2
    hi = lo + check_range
    if lo < start:
        lo, hi = start, start + check_range
    elif hi > end:
        lo, hi = end - check_range, end
    return lo, hi


def select_sharpest(scored: list[tuple[int, float]]) -> int:
    """Frame index with the highest score; ties resolve to the earliest."""
    if not scored:
        raise ValueError("select_sharpest() requires at least one scored frame")
    best_i, best_s = scored[0]
    for i, s in scored[1:]:
        if s > best_s:
            best_i, best_s = i, s
    return best_i


def fit_and_pad(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Aspect-preserving downscale/upscale into width x height with centered black
    padding — mirrors the ffmpeg filter
    `scale=W:H:force_original_aspect_ratio=decrease,pad=W:H:(ow-iw)/2:(oh-ih)/2`.
    """
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, min(width, round(w * scale)))
    new_h = max(1, min(height, round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if new_w == width and new_h == height:
        return resized
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    return cv2.copyMakeBorder(
        resized,
        top, height - new_h - top,
        left, width - new_w - left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness proxy — same metric as FrameFilter.score_frame()."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── extraction loop ───────────────────────────────────────────────────────


def extract_sharpest_frames(video: Path, out_dir: Path, v) -> int:
    """
    Walk `video` sequentially, keep the sharpest frame of each interval, and
    write them to out_dir as %06d.jpg. Returns the number of frames written.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0:
            logger.warning("Native FPS unavailable for %s — assuming 30.", video.name)
            native_fps = 30.0

        interval = interval_length(v.frame_extraction_mode, v.fps, native_fps)
        logger.info(
            "Sharp frame extraction: native %.2f fps, interval %.2f frames, "
            "check range %d.",
            native_fps, interval, v.sharpness_check_range,
        )

        out_count = 0
        cur_k = 0
        cur_start, cur_end = interval_bounds(cur_k, interval)
        lo, hi = candidate_window(cur_start, cur_end, v.sharpness_check_range)
        best_score = -1.0
        best_frame: np.ndarray | None = None
        frame_idx = 0

        def _flush() -> None:
            nonlocal out_count, best_frame, best_score
            if best_frame is not None:
                out_count += 1
                _write_jpeg(out_dir / f"{out_count:06d}.jpg", best_frame, v)
            best_frame = None
            best_score = -1.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            while frame_idx >= cur_end:  # entered a new interval
                _flush()
                cur_k += 1
                cur_start, cur_end = interval_bounds(cur_k, interval)
                lo, hi = candidate_window(cur_start, cur_end, v.sharpness_check_range)
            if lo <= frame_idx < hi:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = laplacian_variance(gray)
                if score > best_score:
                    best_score = score
                    best_frame = frame
            frame_idx += 1

        _flush()  # last (possibly partial) interval
        return out_count
    finally:
        cap.release()


def _write_jpeg(path: Path, frame: np.ndarray, v) -> None:
    img = fit_and_pad(frame, v.resolutionwidth, v.resolutionheight)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"JPEG encode failed for {path.name}")
    # tofile() is unicode-path-safe on Windows (cv2.imwrite is not).
    buf.tofile(str(path))
