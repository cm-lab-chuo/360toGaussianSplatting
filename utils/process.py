"""
Subprocess utilities — run external tools with real-time log streaming.
"""
from __future__ import annotations
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run(
    cmd: list,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
) -> int:
    """
    Run an external command, streaming stdout/stderr to the logger in real time.

    Returns the exit code. Raises subprocess.CalledProcessError on non-zero exit.
    """
    cmd_strs = [str(c) for c in cmd]
    logger.info("$ %s", " ".join(cmd_strs))

    proc = subprocess.Popen(
        cmd_strs,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    def _drain(pipe, level: int) -> None:
        for line in iter(pipe.readline, ""):
            logger.log(level, line.rstrip())
        pipe.close()

    threads = [
        threading.Thread(target=_drain, args=(proc.stdout, logging.INFO), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, logging.WARNING), daemon=True),
    ]
    for t in threads:
        t.start()

    proc.wait()
    for t in threads:
        t.join()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd_strs)

    return proc.returncode


def run_capture(
    cmd: list,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
) -> tuple[int, str]:
    """
    Run a command and return (exit_code, combined stdout+stderr).

    Unlike run(), output is captured instead of streamed, non-zero exit does
    NOT raise, and a missing executable returns (127, "") instead of raising —
    intended for capability probes (e.g. `colmap help`).
    """
    cmd_strs = [str(c) for c in cmd]
    logger.debug("$ %s", " ".join(cmd_strs))
    try:
        proc = subprocess.run(
            cmd_strs,
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, OSError):
        return 127, ""
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
