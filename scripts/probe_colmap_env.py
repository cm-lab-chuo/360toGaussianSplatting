"""
Phase 0 — COLMAP environment probe (docs/360sfm_implementation_plan.md).

Checks the local COLMAP for the capabilities panorama_sfm needs and prints a
report. Run whenever the COLMAP installation changes and paste the result
into docs/colmap_41_environment.md.

Usage:
  python scripts/probe_colmap_env.py [--colmap PATH]
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

# Allow running from the repo root or scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.process import run_capture  # noqa: E402

REQUIRED_VERSION = (4, 1)
NEEDED_COMMANDS = ["panorama_sfm", "view_graph_calibrator", "global_mapper", "mapper"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe local COLMAP for panorama_sfm support")
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable (default: from PATH)")
    args = parser.parse_args()

    code, help_text = run_capture([args.colmap, "help"])
    if code == 127 or not help_text:
        print(f"NG  COLMAP not found: {args.colmap}")
        return 1

    m = re.search(r"COLMAP\s+(\d+)\.(\d+)", help_text)
    version = (int(m.group(1)), int(m.group(2))) if m else None
    version_str = ".".join(map(str, version)) if version else "unknown"
    ok_version = version is not None and version >= REQUIRED_VERSION
    print(f"{'OK ' if ok_version else 'NG '} COLMAP version: {version_str} "
          f"(required for panorama_sfm: >= {'.'.join(map(str, REQUIRED_VERSION))})")

    commands: set[str] = set()
    in_list = False
    for line in help_text.splitlines():
        if line.strip() == "Available commands:":
            in_list = True
            continue
        if in_list and re.fullmatch(r"[a-z0-9_]+", line.strip()):
            commands.add(line.strip())

    for cmd in NEEDED_COMMANDS:
        print(f"{'OK ' if cmd in commands else 'NG '} subcommand: {cmd}")

    try:
        import pycolmap  # type: ignore
        print(f"OK  pycolmap: {pycolmap.__version__}")
    except ImportError:
        print("NG  pycolmap: not installed (needed for the example-script route: pip install pycolmap)")

    print()
    if "panorama_sfm" in commands:
        print("=> --sfm panorama_global / panorama_incremental can use the native subcommand.")
    else:
        print("=> No native panorama_sfm. Set [PanoramaSFMSettings] panorama_script to the")
        print("   pycolmap example (colmap/python/examples/panorama_sfm.py) or install COLMAP 4.1+.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
