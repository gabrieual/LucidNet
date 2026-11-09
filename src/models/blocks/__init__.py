"""Neural blocks for CWD-Net (Blocks 1, 2, 3, and 4)."""

from src.models.blocks.nafblock import LayerNorm2d, SimpleGate, NAFBlock, ChannelAttentionBlock
from src.models.blocks.encoder import ShallowMultiScaleEncoder
from src.models.blocks.rdb_dim import ResidualDenseBlock, DeepIntegrationModule
from src.models.blocks.grum import GatedReferenceUpdateModule
from src.models.blocks.daab import (
    flow_warp,
    LightweightPyramidFlow,
    DeformableAttentionAlignmentBlock,
    RecurrentReferenceAlignmentStage,
    OpticalFlowOnlyAlignmentStage,
)
from src.models.blocks.mtcsa import MultiHeadTemporalChannelAttention, LuckyWeightedAverageFusion
from src.models.blocks.wiener_deconv import MultiChannelWienerDeconv2D, IdentityUpsampleDeconv
from src.models.blocks.twin_decoder import TwinNAFDecoder, SingleStageNAFDecoder

__all__ = [
    "LayerNorm2d",
    "SimpleGate",
    "NAFBlock",
    "ChannelAttentionBlock",
    "ShallowMultiScaleEncoder",
    "ResidualDenseBlock",
    "DeepIntegrationModule",
    "GatedReferenceUpdateModule",
    "flow_warp",
    "LightweightPyramidFlow",
    "DeformableAttentionAlignmentBlock",
    "RecurrentReferenceAlignmentStage",
    "OpticalFlowOnlyAlignmentStage",
    "MultiHeadTemporalChannelAttention",
    "LuckyWeightedAverageFusion",
    "MultiChannelWienerDeconv2D",
    "IdentityUpsampleDeconv",
    "TwinNAFDecoder",
    "SingleStageNAFDecoder",
]
