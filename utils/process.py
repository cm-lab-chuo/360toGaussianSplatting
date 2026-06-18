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
