"""
Abstract base class for all pipeline stages.

To add a new method (e.g. a new masker):
  1. Create a file under pipeline/stages/masking/my_method.py
  2. Subclass Stage, implement `name` and `run()`
  3. Register it in registry.py: MASKING["my_method"] = MyMethod
  4. Run with: python main.py ... --masker my_method
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from config import Config
from pipeline.context import PipelineContext


class Stage(ABC):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name shown in progress output."""
        ...

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this stage; return the updated context."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
