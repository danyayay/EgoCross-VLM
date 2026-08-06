"""Transformer-based multimodal fusion model.

Each enabled modality is projected to a shared dimension ``d_model``, tagged
with a learned modality-type embedding and a shared temporal positional
embedding, then all tokens are concatenated and fed to a Transformer encoder.
A CLS token (optional) or mean pooling over visual tokens is used for the
final classification.
"""

from typing import Optional

import torch
import torch.nn as nn


class TransformerFusion(nn.Module):
    """Transformer fusion model for egocentric multimodal time-series.

    Each enabled modality produces ``T`` tokens per sample. Tokens across all
    modalities are concatenated along the sequence dimension (yielding up to
    ``M * T`` tokens) and processed by a standard Transformer encoder. A
    learnable CLS token is prepended and used as the pooled representation.

    Args:
        visual_dim: Dimensionality of input visual features per timestep.
        pose_dim: Dimensionality of head-pose features (0 to disable).
        gaze_dim: Dimensionality of gaze features (0 to disable).
        motion_dim: Dimensionality of body-frame motion features (0 to disable).
        vel_dim: Dimensionality of velocity features (0 to disable).
        goal_dim: Dimensionality of goal features (0 to disable).
        d_model: Shared embedding dimension inside the Transformer.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feed-forward hidden size inside each encoder layer.
        dropout: Dropout probability applied inside the Transformer.
        n_classes: Number of output intention classes.
        use_cls_token: If ``True`` (default), prepend a learnable CLS token and
            use its output as the pooled representation. Otherwise, mean-pool
            the visual token block.
        max_time_steps: Maximum sequence length ``T`` supported. Determines the
            size of the positional embedding table.

    Example::

        >>> model = TransformerFusion(visual_dim=512, pose_dim=6, gaze_dim=2,
        ...                           d_model=128, nhead=4, num_layers=2,
        ...                           n_classes=2)
        >>> vis  = torch.randn(4, 30, 512)
        >>> pose = torch.randn(4, 30, 6)
        >>> gaze = torch.randn(4, 30, 2)
        >>> logits = model(vis, head_pose=pose, gaze=gaze)
        >>> logits.shape
        torch.Size([4, 2])
    """

    # Fixed ordering used for modality-type embeddings (index → modality)
    _MODALITY_ORDER = ("visual", "pose", "gaze", "motion", "vel", "goal")

    def __init__(
        self,
        visual_dim: int,
        pose_dim: int = 0,
        gaze_dim: int = 0,
        motion_dim: int = 0,
        vel_dim: int = 0,
        goal_dim: int = 0,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_classes: int = 2,
        use_cls_token: bool = True,
        max_time_steps: int = 64,
    ) -> None:
        super().__init__()

        self.visual_dim = visual_dim
        self.pose_dim = pose_dim
        self.gaze_dim = gaze_dim
        self.motion_dim = motion_dim
        self.vel_dim = vel_dim
        self.goal_dim = goal_dim
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.max_time_steps = max_time_steps

        # Per-modality linear projections + LayerNorms
        dims = dict(visual=visual_dim, pose=pose_dim, gaze=gaze_dim,
                    motion=motion_dim, vel=vel_dim, goal=goal_dim)
        for name, dim in dims.items():
            proj = nn.Linear(dim, d_model) if dim > 0 else None
            ln   = nn.LayerNorm(d_model)   if dim > 0 else None
            setattr(self, f"{name}_proj", proj)
            setattr(self, f"{name}_ln",   ln)

        # Shared temporal positional embeddings (one vector per timestep)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_time_steps, d_model))

        # Modality-type embeddings: one row per modality in _MODALITY_ORDER
        self.modality_emb = nn.Parameter(
            torch.randn(len(self._MODALITY_ORDER), d_model) * 0.02
        )

        # Optional CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.pool_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise projection weights with Xavier normal; embeddings with N(0, 0.02)."""
        for name in self._MODALITY_ORDER:
            proj: Optional[nn.Linear] = getattr(self, f"{name}_proj")
            if proj is not None:
                nn.init.xavier_normal_(proj.weight)
                nn.init.zeros_(proj.bias)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.modality_emb, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        visual_feats: torch.Tensor,
        head_pose: Optional[torch.Tensor] = None,
        gaze: Optional[torch.Tensor] = None,
        motion: Optional[torch.Tensor] = None,
        vel: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            visual_feats: ``(B, T, visual_dim)`` — required.
            head_pose: ``(B, T, pose_dim)`` or ``None``.
            gaze: ``(B, T, gaze_dim)`` or ``None``.
            motion: ``(B, T, motion_dim)`` or ``None``.
            vel: ``(B, T, vel_dim)`` or ``None``.
            goal: ``(B, T, goal_dim)`` or ``None``.
            time_mask: ``(B, T)`` boolean mask where ``True`` marks *valid*
                timesteps. When provided, invalid timesteps are masked out in
                the Transformer attention.

        Returns:
            logits: ``(B, n_classes)``

        Raises:
            ValueError: If ``T > max_time_steps``.
        """
        B, T, _ = visual_feats.shape
        if T > self.max_time_steps:
            raise ValueError(f"T={T} exceeds max_time_steps={self.max_time_steps}")

        # Build token list: (modality_index, tensor) pairs for active modalities
        inputs = [
            (0, "visual", visual_feats),
            (1, "pose",   head_pose),
            (2, "gaze",   gaze),
            (3, "motion", motion),
            (4, "vel",    vel),
            (5, "goal",   goal),
        ]

        token_blocks: list[torch.Tensor] = []
        mod_indices: list[int] = []

        for idx, name, tensor in inputs:
            proj: Optional[nn.Linear] = getattr(self, f"{name}_proj")
            ln:   Optional[nn.LayerNorm] = getattr(self, f"{name}_ln")
            if proj is None or tensor is None:
                continue
            t = proj(tensor)                          # (B, T, d_model)
            if ln is not None:
                t = ln(t)
            t = t + self.pos_emb[:, :T, :]           # add temporal position
            token_blocks.append(t)
            mod_indices.append(idx)

        # Add modality-type embeddings
        x = torch.cat(token_blocks, dim=1)            # (B, M*T, d_model)
        if len(mod_indices) != 1:
            type_emb = torch.cat(
                [self.modality_emb[i:i+1].unsqueeze(1).expand(1, T, -1) for i in mod_indices],
                dim=1,
            ).expand(B, -1, -1)                           # (B, M*T, d_model)
            x = x + type_emb

        # Prepend CLS token
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        # Build key-padding mask from time_mask if provided
        src_key_padding_mask = None
        if time_mask is not None:
            pad = torch.cat([~time_mask] * len(mod_indices), dim=1)  # (B, M*T)
            if self.cls_token is not None:
                pad = torch.cat(
                    [torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], dim=1
                )
            src_key_padding_mask = pad

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask) # (B, M*(T+1), d_model)

        # Pool
        pooled = x[:, 0, :] if self.cls_token is not None else x[:, :T, :].mean(dim=1) # (B, d_model)
        pooled = self.pool_norm(pooled) # (B, d_model)
        return self.classifier(pooled) # (B, n_classes)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T = 2, 30
    model = TransformerFusion(
        visual_dim=512, pose_dim=6, gaze_dim=2, motion_dim=2, vel_dim=2, goal_dim=3,
        d_model=128, nhead=4, num_layers=2, n_classes=2, max_time_steps=64,
    ).to(device)
    vis  = torch.randn(B, T, 512, device=device)
    pose = torch.randn(B, T, 6,   device=device)
    gaze = torch.randn(B, T, 2,   device=device)
    mot  = torch.randn(B, T, 2,   device=device)
    vel  = torch.randn(B, T, 2,   device=device)
    goal = torch.randn(B, T, 3,   device=device)
    logits = model(vis, head_pose=pose, gaze=gaze, motion=mot, vel=vel, goal=goal)
    print("logits.shape:", logits.shape)  # (2, 2)
