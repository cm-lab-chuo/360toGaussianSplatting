"""
Stage 4b — Passthrough masker (no masking).

Simply skips masking; downstream stages will use the cubemap/frames directly.
Useful when the scene has no dynamic objects, or when you want to run
SfM before deciding on a masking strategy.
"""
from __future__ import annotations
import logging

from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class PassthroughMasker(Stage):

    @property
    def name(self) -> str:
        return "Masking (none)"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("No masking applied.")
        return ctx
