"""
Pipeline orchestrator — runs a list of stages in order.
"""
from __future__ import annotations
import logging
import time
from typing import Sequence

from pipeline.base import Stage
from pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, stages: Sequence[Stage]) -> None:
        self.stages = list(stages)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("Pipeline: %d stage(s) — %s",
                    len(self.stages),
                    " → ".join(s.name for s in self.stages))

        for stage in self.stages:
            logger.info("─── [%s] starting ───", stage.name)
            t0 = time.perf_counter()
            ctx = stage.run(ctx)
            elapsed = time.perf_counter() - t0
            logger.info("─── [%s] done (%.1f s) ───", stage.name, elapsed)

        logger.info("Pipeline complete.")
        return ctx
