"""
Stage registry — the single place to add or swap pipeline methods.

To add a new method:
  1. Create a file under the appropriate pipeline/stages/ subdirectory
  2. Implement the Stage interface (subclass Stage, define name + run())
  3. Add an entry to the relevant dict below

CLI selects implementations via --masker / --sfm / --frame-filter flags.
"""
from pipeline.stages.preprocessing.ffmpeg_extractor import FFmpegExtractor
from pipeline.stages.preprocessing.frame_filter import FrameFilter
from pipeline.stages.preprocessing.cubemap_splitter import CubemapSplitter

from pipeline.stages.masking.automasker import AutoMaskerStage
from pipeline.stages.masking.passthrough import PassthroughMasker
from pipeline.stages.masking.pregenerated import PregeneratedMasker

from functools import partial

from pipeline.stages.sfm.spheresfm import SphereSFMStage
from pipeline.stages.sfm.colmap import COLMAPStage
from pipeline.stages.sfm.colmap_panorama import PanoramaSFMStage
from pipeline.stages.sfm.realitycapture import RealityCaptureStage


# ── Preprocessing ─────────────────────────────────────────────────────────
EXTRACTOR: dict = {
    "ffmpeg": FFmpegExtractor,
    # "custom": MyCustomExtractor,
}

FRAME_FILTER: dict = {
    "laplacian": FrameFilter,   # default sharpness metric
    # "blur_score": BlurScoreFilter,
}

CUBEMAP_SPLITTER: dict = {
    "perspective": CubemapSplitter,  # equirect → N perspective crops
    # "equirect": EquirectPassthrough,  # skip splitting, use full equirect
}

# ── Masking ───────────────────────────────────────────────────────────────
MASKING: dict = {
    "automasker":   AutoMaskerStage,     # GroundingDINO + SAM2
    "none":         PassthroughMasker,   # no masking
    "pregenerated": PregeneratedMasker,  # load existing mask files
    # "sam2_direct": SAM2DirectMasker,   # future: direct SAM2 without AutoMasker wrapper
}

# ── SfM / Camera Alignment ────────────────────────────────────────────────
SFM: dict = {
    "spheresfm":      SphereSFMStage,      # 360°-aware COLMAP variant for spherical input
    "colmap":         COLMAPStage,         # standard COLMAP
    # COLMAP 4.1 panorama_sfm: ERP frames → virtual rig views → global/incremental mapper
    "panorama_global":      partial(PanoramaSFMStage, mapper="global"),
    "panorama_incremental": partial(PanoramaSFMStage, mapper="incremental"),
    "realitycapture": RealityCaptureStage, # Epic RealityScan
    # "hloc":         HLocStage,           # future: hierarchical localization
    # "instant_splat": InstantSplatStage,  # future: feed-forward SfM
}
