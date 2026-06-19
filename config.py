"""
Configuration management for the 360Gaussian pipeline.

Loads pipeline parameters from an INI configuration file.
All pipeline parameters are accessible as typed dataclass attributes.
"""
from __future__ import annotations
import configparser
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoSettings:
    fps: float = 1.0
    frame_extraction_mode: str = "seconds"  # "seconds" | "frames"
    splits: int = 8                         # number of perspective crops per equirect frame
    resolution: int = 1920
    resolutionwidth: int = 1920
    resolutionheight: int = 1920
    fovvalue: float = 90.0                  # FOV of each perspective crop (degrees)
    processingmode: str = "hybrid"          # "hybrid" | "equirect" | "cubemap"
    usegpu: bool = True
    useframefilter: bool = False
    filtermethod: str = "best_n"            # "best_n" | "threshold"
    framestokeep: str = "50%"              # "50%" or integer
    sharpnessthreshold: float = 100.0
    usemultitilt: bool = False
    tiltangle2: float = 30.0
    tiltangle3: float = -30.0
    autocontinue: bool = True


@dataclass
class AutoMaskerPaths:
    automaskerpath: Path = field(default_factory=lambda: Path(""))
    configfile: Path = field(default_factory=lambda: Path(""))
    dinoconfig: Path = field(default_factory=lambda: Path(""))
    dinocheckpoint: Path = field(default_factory=lambda: Path(""))
    samconfig: Path = field(default_factory=lambda: Path(""))
    samcheckpoint: Path = field(default_factory=lambda: Path(""))


@dataclass
class AutoMaskerSettings:
    useautomasker: bool = True
    keywords: str = "person.sky"           # dot-separated detection targets
    boxthreshold: float = 0.35
    textthreshold: float = 0.25
    exportmaskonly: bool = True
    exporttransparent: bool = True
    exportcolored: bool = False
    invertmask: bool = True
    pregenerated_masks_path: Path = field(default_factory=lambda: Path(""))
    maskexpand: float = 5.0
    custom_mask_only: bool = False
    custom_mask_path: Path = field(default_factory=lambda: Path(""))


@dataclass
class AlignmentSettings:
    usemarkers: bool = False
    markerdistance: float = 0.5
    training_method: str = "no_training"   # "no_training"|"postshot"|"brush"|"lichtfeld"
    alignment_method_brush: str = "realityscan"
    alignment_method_postshot: str = "postshot"
    alignment_method_no_training: str = "spheresfm"
    alignment_method_lichtfeld: str = "spheresfm"
    export_stability_wait_time: float = 3.0
    max_export_stability_checks: int = 10
    graceful_termination_timeout: float = 10.0
    markertype: str = "1x12"              # "1x12" | "16h5"


@dataclass
class MaskSettings:
    # Whether masks (from any source) are fed into the SfM feature extractor.
    apply_masks_to_sfm: bool = True
    # Masks produced by a separate/external program. Used when no in-pipeline
    # masker ran (e.g. --masker none) or to override the masker's output.
    external_mask_dir: Path = field(default_factory=lambda: Path(""))
    # COLMAP ignores BLACK (0) pixels. Detected/dynamic regions must therefore be
    # black in the COLMAP mask. AutoMasker outputs removed regions as WHITE, so
    # inversion is needed by default. Flip to False for sources that already use
    # COLMAP polarity (white = keep).
    mask_invert: bool = True
    # Comma-separated filename-stem suffixes to try when matching a raw mask to an
    # image (e.g. AutoMasker writes "<stem>.mask.png"). "" (no suffix) is always
    # also tried.
    mask_suffixes: str = ".mask,_mask,_masked"
    # Grayscale threshold for binarizing the raw mask (0-255).
    mask_threshold: int = 127


@dataclass
class SphereSFMSettings:
    max_num_features: int = 32768
    first_octave: int = 0
    peak_threshold: float = 0.004
    edge_threshold: float = 12.0
    max_num_matches: int = 65536
    matcher_type: str = "sequential"
    estimate_affine_shape: bool = True
    domain_size_pooling: bool = True
    guided_matching: bool = False
    ba_local_max_iterations: int = 40
    ba_global_max_iterations: int = 100
    ba_global_max_refinements: int = 2
    filter_max_reproj_error: float = 2.5
    filter_min_tri_angle: float = 1.75
    abs_pose_min_num_inliers: int = 50
    run_bundle_adjuster: bool = True
    realign_cubemaps: bool = True
    cubemap_refine_focal: bool = False


@dataclass
class ToolPaths:
    # External executables. Leave blank to resolve from PATH.
    ffmpeg: Path = field(default_factory=lambda: Path(""))
    colmap_sphere: Path = field(default_factory=lambda: Path(""))


@dataclass
class COLMAPPaths:
    colmappath: Path = field(default_factory=lambda: Path(""))


@dataclass
class PostShotPaths:
    realitycapturepath: Path = field(
        default_factory=lambda: Path("C:/Program Files/Epic Games/RealityScan_2.1/RealityScan.exe")
    )
    postshotpath: Path = field(
        default_factory=lambda: Path("C:/Program Files/Jawset Postshot/bin/postshot-cli.exe")
    )
    settingsfolder: Path = field(default_factory=lambda: Path(""))


@dataclass
class PostShotSettings:
    profile: str = "Splat MCMC"
    maximagesize: int = 0
    trainsteps: int = 25
    maxsplats: int = 3000
    showtrainerror: bool = True
    antialiasing: bool = False
    exportply: bool = True


@dataclass
class BrushSettings:
    brush_path: Path = field(default_factory=lambda: Path(""))
    total_steps: int = 25
    max_splats: int = 3000
    sh_degree: int = 3
    max_resolution: int = 1920
    export_every: int = 5000


@dataclass
class LichtfeldSettings:
    lichtfeld_path: Path = field(default_factory=lambda: Path(""))
    iter: int = 30
    sh_degree: int = 3
    max_width: int = 3840
    strategy: str = "mrnf"
    max_cap: int = 3000
    steps_scaler: float = 1.0


@dataclass
class Config:
    postshot_paths: PostShotPaths = field(default_factory=PostShotPaths)
    tool_paths: ToolPaths = field(default_factory=ToolPaths)
    colmap_paths: COLMAPPaths = field(default_factory=COLMAPPaths)
    automasker_paths: AutoMaskerPaths = field(default_factory=AutoMaskerPaths)
    video: VideoSettings = field(default_factory=VideoSettings)
    automasker: AutoMaskerSettings = field(default_factory=AutoMaskerSettings)
    mask: MaskSettings = field(default_factory=MaskSettings)
    alignment: AlignmentSettings = field(default_factory=AlignmentSettings)
    spheresfm: SphereSFMSettings = field(default_factory=SphereSFMSettings)
    postshot: PostShotSettings = field(default_factory=PostShotSettings)
    brush: BrushSettings = field(default_factory=BrushSettings)
    lichtfeld: LichtfeldSettings = field(default_factory=LichtfeldSettings)

    # ------------------------------------------------------------------ #
    #  INI loader                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_ini(cls, path: Path) -> Config:
        # RawConfigParser avoids % interpolation issues.
        # inline_comment_prefixes lets us write "key = value  ; explanation" in the
        # INI and have the comment stripped (none of our values contain ';').
        p = configparser.RawConfigParser(inline_comment_prefixes=(";",))
        p.read(str(path), encoding="utf-8")
        cfg = cls()

        def _s(sec: str, key: str, default: str = "") -> str:
            return p[sec].get(key, default) if sec in p else default

        def _b(sec: str, key: str, default: bool = False) -> bool:
            return p[sec].getboolean(key, default) if sec in p else default

        def _f(sec: str, key: str, default: float = 0.0) -> float:
            return float(_s(sec, key, str(default)))

        def _i(sec: str, key: str, default: int = 0) -> int:
            return int(_s(sec, key, str(default)))

        def _p(sec: str, key: str, default: str = "") -> Path:
            return Path(_s(sec, key, default))

        # PostShotPaths
        cfg.postshot_paths.realitycapturepath = _p("PostShotPaths", "realitycapturepath")
        cfg.postshot_paths.postshotpath = _p("PostShotPaths", "postshotpath")
        cfg.postshot_paths.settingsfolder = _p("PostShotPaths", "settingsfolder")

        # VideoSettings
        cfg.video.fps = _f("VideoSettings", "fps", 1.0)
        cfg.video.frame_extraction_mode = _s("VideoSettings", "frame_extraction_mode", "seconds")
        cfg.video.splits = _i("VideoSettings", "splits", 8)
        cfg.video.resolution = _i("VideoSettings", "resolution", 1920)
        cfg.video.resolutionwidth = _i("VideoSettings", "resolutionwidth", 1920)
        cfg.video.resolutionheight = _i("VideoSettings", "resolutionheight", 1920)
        cfg.video.fovvalue = _f("VideoSettings", "fovvalue", 90.0)
        cfg.video.processingmode = _s("VideoSettings", "processingmode", "hybrid")
        cfg.video.usegpu = _b("VideoSettings", "usegpu", True)
        cfg.video.useframefilter = _b("VideoSettings", "useframefilter", False)
        cfg.video.filtermethod = _s("VideoSettings", "filtermethod", "best_n")
        cfg.video.framestokeep = _s("VideoSettings", "framestokeep", "50%")
        cfg.video.sharpnessthreshold = _f("VideoSettings", "sharpnessthreshold", 100.0)
        cfg.video.usemultitilt = _b("VideoSettings", "usemultitilt", False)
        cfg.video.tiltangle2 = _f("VideoSettings", "tiltangle2", 30.0)
        cfg.video.tiltangle3 = _f("VideoSettings", "tiltangle3", -30.0)
        cfg.video.autocontinue = _b("VideoSettings", "autocontinue", True)

        # AutoMaskerPaths
        cfg.automasker_paths.automaskerpath = _p("AutoMaskerPaths", "automaskerpath")
        cfg.automasker_paths.configfile = _p("AutoMaskerPaths", "configfile")
        cfg.automasker_paths.dinoconfig = _p("AutoMaskerPaths", "dinoconfig")
        cfg.automasker_paths.dinocheckpoint = _p("AutoMaskerPaths", "dinocheckpoint")
        cfg.automasker_paths.samconfig = _p("AutoMaskerPaths", "samconfig")
        cfg.automasker_paths.samcheckpoint = _p("AutoMaskerPaths", "samcheckpoint")

        # AutoMaskerSettings
        cfg.automasker.useautomasker = _b("AutoMaskerSettings", "useautomasker", True)
        cfg.automasker.keywords = _s("AutoMaskerSettings", "keywords", "person.sky")
        cfg.automasker.boxthreshold = _f("AutoMaskerSettings", "boxthreshold", 0.35)
        cfg.automasker.textthreshold = _f("AutoMaskerSettings", "textthreshold", 0.25)
        cfg.automasker.exportmaskonly = _b("AutoMaskerSettings", "exportmaskonly", True)
        cfg.automasker.exporttransparent = _b("AutoMaskerSettings", "exporttransparent", True)
        cfg.automasker.exportcolored = _b("AutoMaskerSettings", "exportcolored", False)
        cfg.automasker.invertmask = _b("AutoMaskerSettings", "invertmask", True)
        cfg.automasker.pregenerated_masks_path = _p("AutoMaskerSettings", "pregenerated_masks_path")
        cfg.automasker.maskexpand = _f("AutoMaskerSettings", "maskexpand", 5.0)
        cfg.automasker.custom_mask_only = _b("AutoMaskerSettings", "custom_mask_only", False)
        cfg.automasker.custom_mask_path = _p("AutoMaskerSettings", "custom_mask_path")

        # ToolPaths
        cfg.tool_paths.ffmpeg = _p("ToolPaths", "ffmpeg")
        cfg.tool_paths.colmap_sphere = _p("ToolPaths", "colmap_sphere")

        # MaskSettings
        cfg.mask.apply_masks_to_sfm = _b("MaskSettings", "apply_masks_to_sfm", True)
        cfg.mask.external_mask_dir = _p("MaskSettings", "external_mask_dir")
        cfg.mask.mask_invert = _b("MaskSettings", "mask_invert", False)
        cfg.mask.mask_suffixes = _s("MaskSettings", "mask_suffixes", ".mask,_mask,_masked")
        cfg.mask.mask_threshold = _i("MaskSettings", "mask_threshold", 127)

        # COLMAPPaths
        cfg.colmap_paths.colmappath = _p("COLMAPPaths", "colmappath")

        # AlignmentSettings
        cfg.alignment.usemarkers = _b("AlignmentSettings", "usemarkers", False)
        cfg.alignment.markerdistance = _f("AlignmentSettings", "markerdistance", 0.5)
        cfg.alignment.training_method = _s("AlignmentSettings", "training_method", "no_training")
        cfg.alignment.alignment_method_brush = _s("AlignmentSettings", "alignment_method_brush", "realityscan")
        cfg.alignment.alignment_method_postshot = _s("AlignmentSettings", "alignment_method_postshot", "postshot")
        cfg.alignment.alignment_method_no_training = _s("AlignmentSettings", "alignment_method_no_training", "spheresfm")
        cfg.alignment.alignment_method_lichtfeld = _s("AlignmentSettings", "alignment_method_lichtfeld", "spheresfm")
        cfg.alignment.export_stability_wait_time = _f("AlignmentSettings", "export_stability_wait_time", 3.0)
        cfg.alignment.max_export_stability_checks = _i("AlignmentSettings", "max_export_stability_checks", 10)
        cfg.alignment.graceful_termination_timeout = _f("AlignmentSettings", "graceful_termination_timeout", 10.0)
        cfg.alignment.markertype = _s("AlignmentSettings", "markertype", "1x12")

        # SphereSFMSettings
        cfg.spheresfm.max_num_features = _i("SphereSFMSettings", "max_num_features", 32768)
        cfg.spheresfm.first_octave = _i("SphereSFMSettings", "first_octave", 0)
        cfg.spheresfm.peak_threshold = _f("SphereSFMSettings", "peak_threshold", 0.004)
        cfg.spheresfm.edge_threshold = _f("SphereSFMSettings", "edge_threshold", 12.0)
        cfg.spheresfm.max_num_matches = _i("SphereSFMSettings", "max_num_matches", 65536)
        cfg.spheresfm.matcher_type = _s("SphereSFMSettings", "matcher_type", "sequential")
        cfg.spheresfm.estimate_affine_shape = _b("SphereSFMSettings", "estimate_affine_shape", True)
        cfg.spheresfm.domain_size_pooling = _b("SphereSFMSettings", "domain_size_pooling", True)
        cfg.spheresfm.guided_matching = _b("SphereSFMSettings", "guided_matching", False)
        cfg.spheresfm.ba_local_max_iterations = _i("SphereSFMSettings", "ba_local_max_iterations", 40)
        cfg.spheresfm.ba_global_max_iterations = _i("SphereSFMSettings", "ba_global_max_iterations", 100)
        cfg.spheresfm.ba_global_max_refinements = _i("SphereSFMSettings", "ba_global_max_refinements", 2)
        cfg.spheresfm.filter_max_reproj_error = _f("SphereSFMSettings", "filter_max_reproj_error", 2.5)
        cfg.spheresfm.filter_min_tri_angle = _f("SphereSFMSettings", "filter_min_tri_angle", 1.75)
        cfg.spheresfm.abs_pose_min_num_inliers = _i("SphereSFMSettings", "abs_pose_min_num_inliers", 50)
        cfg.spheresfm.run_bundle_adjuster = _b("SphereSFMSettings", "run_bundle_adjuster", True)
        cfg.spheresfm.realign_cubemaps = _b("SphereSFMSettings", "realign_cubemaps", True)
        cfg.spheresfm.cubemap_refine_focal = _b("SphereSFMSettings", "cubemap_refine_focal", False)

        # PostShotSettings
        cfg.postshot.profile = _s("PostShotSettings", "profile", "Splat MCMC")
        cfg.postshot.maximagesize = _i("PostShotSettings", "maximagesize", 0)
        cfg.postshot.trainsteps = _i("PostShotSettings", "trainsteps", 25)
        cfg.postshot.maxsplats = _i("PostShotSettings", "maxsplats", 3000)
        cfg.postshot.showtrainerror = _b("PostShotSettings", "showtrainerror", True)
        cfg.postshot.antialiasing = _b("PostShotSettings", "antialiasing", False)
        cfg.postshot.exportply = _b("PostShotSettings", "exportply", True)

        # BrushSettings
        cfg.brush.brush_path = _p("BrushSettings", "brush_path")
        cfg.brush.total_steps = _i("BrushSettings", "total_steps", 25)
        cfg.brush.max_splats = _i("BrushSettings", "max_splats", 3000)
        cfg.brush.sh_degree = _i("BrushSettings", "sh_degree", 3)
        cfg.brush.max_resolution = _i("BrushSettings", "max_resolution", 1920)
        cfg.brush.export_every = _i("BrushSettings", "export_every", 5000)

        # LichtfeldSettings
        cfg.lichtfeld.lichtfeld_path = _p("LichtfeldSettings", "lichtfeld_path")
        cfg.lichtfeld.iter = _i("LichtfeldSettings", "iter", 30)
        cfg.lichtfeld.sh_degree = _i("LichtfeldSettings", "sh_degree", 3)
        cfg.lichtfeld.max_width = _i("LichtfeldSettings", "max_width", 3840)
        cfg.lichtfeld.strategy = _s("LichtfeldSettings", "strategy", "mrnf")
        cfg.lichtfeld.max_cap = _i("LichtfeldSettings", "max_cap", 3000)
        cfg.lichtfeld.steps_scaler = _f("LichtfeldSettings", "steps_scaler", 1.0)

        return cfg
