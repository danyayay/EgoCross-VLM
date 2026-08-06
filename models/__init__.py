"""Model architectures for pedestrian intention prediction.

Available models
----------------
backbone
    :class:`CNNBackbone`, :class:`CLIPBackbone` — per-frame feature extractors.
lstm_fusion
    :class:`LSTMFusion` — LSTM that concatenates all modalities along the feature dim.
transformer_fusion
    :class:`TransformerFusion` — Transformer that treats each modality as a token stream.
cross_attention_fusion
    :class:`CrossAttentionFusion` — Hierarchical model: visual self-attention then
    cross-attention to auxiliary modalities with adaptive goal conditioning.
"""

from models.backbone import CLIPBackbone, CNNBackbone
from models.cross_attention_fusion import CrossAttentionFusion
from models.lstm_fusion import LSTMFusion
from models.transformer_fusion import TransformerFusion

__all__ = [
    "CNNBackbone",
    "CLIPBackbone",
    "LSTMFusion",
    "TransformerFusion",
    "CrossAttentionFusion",
]
