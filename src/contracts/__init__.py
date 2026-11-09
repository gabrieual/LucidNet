"""Contracts package exposing typed batch containers and SOLID interfaces."""

from src.contracts.batch import ModelOutput, PriorOutput, VideoBatch
from src.contracts.interfaces import (
    BaseAlignmentBlock,
    BaseDeconvBlock,
    BaseMultiScaleEncoder,
    BasePriorExtractor,
    BaseTemporalFusionBlock,
    BaseTwinDecoder,
    InferenceModel,
    TrainingLossWrapper,
)

__all__ = [
    "VideoBatch",
    "PriorOutput",
    "ModelOutput",
    "BasePriorExtractor",
    "BaseMultiScaleEncoder",
    "BaseAlignmentBlock",
    "BaseTemporalFusionBlock",
    "BaseDeconvBlock",
    "BaseTwinDecoder",
    "InferenceModel",
    "TrainingLossWrapper",
]
