"""
360Gaussian Pipeline — research-oriented CLI.

Usage examples:

  # Full pipeline (default: SphereSFM + AutoMasker)
  python main.py video.mp4 output/ --config config/default.ini

  # Swap SfM method to COLMAP
  python main.py video.mp4 output/ --sfm colmap

  # Skip masking
  python main.py video.mp4 output/ --masker none

  # Resume: skip extraction if frames already exist (--skip extraction)
  python main.py video.mp4 output/ --skip extraction

  # Run only cubemap split + masking (skip extraction and SfM)
  python main.py video.mp4 output/ --skip extraction,filter --stop-after masking

  # Override a config parameter on the command line
  python main.py video.mp4 output/ --set VideoSettings.splits=12

Available SfM methods:    spheresfm (default), colmap, realitycapture
Available maskers:        automasker (default), none, pregenerated

Output layout:
  output/
    frames/     — raw equirectangular frames
    cubemap/    — perspective crops (inputs to SfM)
    masked/     — masked images (if masking enabled)
    sparse/     — COLMAP sparse reconstruction (cameras + poses + sparse pointcloud)
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

from config import Config
from pipeline.context import PipelineContext
from pipeline.orchestrator import Pipeline
from registry import MASKING, SFM, FRAME_FILTER


# ── logging setup ──────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    # On a Japanese Windows console the default stream encoding is cp932; non-ASCII
    # characters in log messages (→, —, file paths) raise UnicodeEncodeError when
    # output is redirected to a file or pipe. Force UTF-8 with a safe fallback so
    # logging never crashes the pipeline regardless of console codepage.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (ValueError, OSError):
                pass

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── pipeline builder ───────────────────────────────────────────────────────

STAGE_ORDER = ["extraction", "filter", "cubemap", "masking", "sfm"]

def build_pipeline(
    cfg: Config,
    masker: str,
    sfm: str,
    skip: set[str],
    stop_after: str | None,
) -> Pipeline:
    """
    Assemble the stage list from the registry based on CLI flags.

    To add a new stage to the default pipeline, import it here and append.
    """
    from pipeline.stages.preprocessing.ffmpeg_extractor import FFmpegExtractor
    from pipeline.stages.preprocessing.frame_filter import FrameFilter
    from pipeline.stages.preprocessing.cubemap_splitter import CubemapSplitter

    all_stages = [
        ("extraction", FFmpegExtractor(cfg)),
        ("filter",     FrameFilter(cfg)),
        ("cubemap",    CubemapSplitter(cfg)),
        ("masking",    MASKING[masker](cfg)),
        ("sfm",        SFM[sfm](cfg)),
    ]

    stages = []
    for key, stage in all_stages:
        if key in skip:
            continue
        stages.append(stage)
        if stop_after and key == stop_after:
            break

    return Pipeline(stages)


# ── config override helper ─────────────────────────────────────────────────

def _apply_overrides(cfg: Config, overrides: list[str]) -> None:
    """
    Apply --set Section.key=value overrides to an already-loaded Config.

    Example: --set VideoSettings.splits=12
    """
    for override in overrides:
        if "=" not in override or "." not in override:
            raise ValueError(f"Invalid --set format (expected Section.key=value): {override!r}")
        key_path, _, value = override.partition("=")
        section, _, attr = key_path.partition(".")
        section_map = {
            "VideoSettings": cfg.video,
            "AutoMaskerSettings": cfg.automasker,
            "AutoMaskerPaths": cfg.automasker_paths,
            "AlignmentSettings": cfg.alignment,
            "SphereSFMSettings": cfg.spheresfm,
            "PostShotSettings": cfg.postshot,
            "BrushSettings": cfg.brush,
            "LichtfeldSettings": cfg.lichtfeld,
            "COLMAPPaths": cfg.colmap_paths,
            "PostShotPaths": cfg.postshot_paths,
        }
        if section not in section_map:
            raise ValueError(f"Unknown config section: {section!r}")
        target = section_map[section]
        if not hasattr(target, attr):
            raise ValueError(f"Unknown config key: {section}.{attr}")
        current = getattr(target, attr)
        # Cast to the same type as the current value
        if isinstance(current, bool):
            setattr(target, attr, value.lower() in ("true", "1", "yes"))
        elif isinstance(current, int):
            setattr(target, attr, int(value))
        elif isinstance(current, float):
            setattr(target, attr, float(value))
        elif isinstance(current, Path):
            setattr(target, attr, Path(value))
        else:
            setattr(target, attr, value)
        logging.getLogger(__name__).info("Config override: %s.%s = %s", section, attr, value)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="360Gaussian Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Input video file or folder of images")
    parser.add_argument("output", type=Path, help="Output / working directory")
    parser.add_argument(
        "--config", type=Path, default=Path("config/default.ini"),
        help="Path to pipeline config INI file (default: config/default.ini)",
    )
    parser.add_argument(
        "--masker", default="automasker",
        choices=list(MASKING.keys()),
        help="Masking method (default: automasker)",
    )
    parser.add_argument(
        "--sfm", default="spheresfm",
        choices=list(SFM.keys()),
        help="SfM / camera alignment method (default: spheresfm)",
    )
    parser.add_argument(
        "--skip", default="",
        metavar="STAGE[,STAGE…]",
        help=f"Comma-separated stages to skip. Choices: {','.join(STAGE_ORDER)}",
    )
    parser.add_argument(
        "--stop-after",
        choices=STAGE_ORDER,
        metavar="STAGE",
        help="Stop the pipeline after this stage",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="Section.key=value",
        help="Override a config value, e.g. --set VideoSettings.splits=12",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load config
    cfg_path = args.config
    if not cfg_path.exists():
        logger.warning("Config not found at %s — using defaults.", cfg_path)
        cfg = Config()
    else:
        cfg = Config.from_ini(cfg_path)
        logger.info("Loaded config: %s", cfg_path)

    # Apply --set overrides
    if args.overrides:
        _apply_overrides(cfg, args.overrides)

    # Parse --skip
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    # Build pipeline
    pipeline = build_pipeline(cfg, args.masker, args.sfm, skip, args.stop_after)

    # Build context
    args.output.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(input_path=args.input, work_dir=args.output)

    # Run
    logger.info("Input:  %s", args.input)
    logger.info("Output: %s", args.output)
    pipeline.run(ctx)

    logger.info("Done.  Results in: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
