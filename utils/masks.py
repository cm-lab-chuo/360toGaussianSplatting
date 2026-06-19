"""
Mask handling — normalize masks from ANY source into COLMAP's convention.

COLMAP feature extraction takes per-image masks via --ImageReader.mask_path.
Convention:
  * For an image  <image_path>/foo.jpg  the mask is  <mask_path>/foo.jpg.png
  * Pixels with value 0 (black) are IGNORED during feature extraction.

Masks may be produced by the in-pipeline AutoMasker stage OR by a completely
separate program. This module bridges whatever naming/polarity the source uses
to the strict convention COLMAP expects, so the SfM stage stays source-agnostic.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _candidate_mask_names(image: Path, suffixes: list[str]) -> list[str]:
    """Filenames to look for in the raw mask dir, most specific first."""
    stem, name = image.stem, image.name
    names: list[str] = []
    for suf in suffixes + [""]:               # "" → no suffix, always tried last
        names += [
            f"{stem}{suf}.png", f"{stem}{suf}.jpg",   # <stem>.mask.png, <stem>.png
            f"{name}{suf}.png",                        # <name>.mask.png
            f"{name}.png",                             # COLMAP-style <name>.png
        ]
    # de-dup while preserving order
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def build_colmap_masks(
    images_dir: Path,
    raw_mask_dir: Path,
    dest_dir: Path,
    *,
    suffixes: list[str],
    invert: bool,
    threshold: int,
) -> tuple[int, int]:
    """
    For every image in images_dir, find its raw mask, binarize/normalize it, and
    write dest_dir/<image_name>.png in COLMAP convention (black = ignored).

    Returns (matched, missing).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)

    matched = missing = 0
    for img_path in images:
        mask_path: Optional[Path] = None
        for cand in _candidate_mask_names(img_path, suffixes):
            p = raw_mask_dir / cand
            if p.is_file():
                mask_path = p
                break
        if mask_path is None:
            missing += 1
            continue

        m = _load_mask(mask_path)
        if m is None:
            missing += 1
            continue

        # Match the image dimensions (COLMAP requires identical size).
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None and m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        binary = (m >= threshold).astype(np.uint8) * 255
        if invert:
            binary = 255 - binary

        cv2.imwrite(str(dest_dir / f"{img_path.name}.png"), binary)
        matched += 1

    return matched, missing


def resolve_colmap_mask_path(ctx, cfg) -> Optional[Path]:
    """
    Resolve masks for SfM from any source and return a COLMAP-ready mask dir
    (or None if masking should not be applied).

    Source precedence:
      1. [MaskSettings] external_mask_dir  (masks from a separate program)
      2. ctx.mask_dir                      (masks produced by the masker stage)

    The external dir wins so a user can override the in-pipeline masker with
    externally-computed masks without changing the masker.
    """
    ms = cfg.mask
    if not ms.apply_masks_to_sfm:
        return None

    def _usable(d) -> bool:
        return bool(d) and str(d) not in ("", ".") and Path(d).exists() and any(Path(d).iterdir())

    raw = None
    if _usable(ms.external_mask_dir):
        raw = Path(ms.external_mask_dir)
        logger.info("Using external masks: %s", raw)
    elif _usable(ctx.mask_dir):
        raw = Path(ctx.mask_dir)
        logger.info("Using masker-produced masks: %s", raw)
    else:
        return None

    images_dir = ctx.images_for_sfm()
    dest = ctx.work_dir / "colmap_masks"
    suffixes = [s for s in (x.strip() for x in ms.mask_suffixes.split(",")) if s]
    matched, missing = build_colmap_masks(
        images_dir, raw, dest,
        suffixes=suffixes, invert=ms.mask_invert, threshold=ms.mask_threshold,
    )
    if matched == 0:
        logger.warning(
            "No masks in %s matched images in %s — running SfM without masks. "
            "Check [MaskSettings] mask_suffixes / the mask filenames.",
            raw, images_dir,
        )
        return None
    logger.info("Prepared %d COLMAP masks (%d images without a mask) → %s",
                matched, missing, dest)
    return dest


def _load_mask(path: Path) -> Optional[np.ndarray]:
    """Load a mask as single-channel uint8. Uses alpha channel if present
    (transparent PNGs encode the mask in alpha)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        logger.warning("Could not read mask %s", path)
        return None
    if raw.ndim == 3 and raw.shape[2] == 4:      # BGRA → use alpha as the mask
        return raw[:, :, 3]
    if raw.ndim == 3:                            # BGR → grayscale
        return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    return raw
