"""Hierarchical cross-attention fusion model.

Architecture overview:
1. Self-attend over the visual token sequence to build temporal visual context.
2. Cross-attend from the visual context to each auxiliary modality (pose,
   motion, gaze) and accumulate their contributions via residual addition.
3. Add learned temporal embeddings.
4. Optionally apply adaptive goal conditioning: a per-timestep gate mixes the
   fused representation with the projected goal embedding.
5. A second self-attention pass over the conditioned sequence, then mean pool
   and classify.
"""

from typing import Optional

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Hierarchical cross-attention fusion model for egocentric time-series.

    The visual stream is self-attended first, then cross-attended against each
    auxiliary modality. A learned gate blends the fused representation with
    goal-conditioned embeddings before final classification.

    Args:
        visual_dim: Dimensionality of visual features per timestep.
        pose_dim: Dimensionality of head-pose features (0 to disable).
        gaze_dim: Dimensionality of gaze features (0 to disable).
        motion_dim: Dimensionality of body-frame motion features (0 to disable).
        vel_dim: Accepted for API consistency; not used internally.
        goal_dim: Dimensionality of goal features (0 to disable goal conditioning).
        d_model: Shared hidden dimension for all projections and attention layers.
        nhead: Number of attention heads (used for both self- and cross-attention).
        num_layers: Number of Transformer encoder layers for self-attention.
        dim_feedforward: Feed-forward hidden size inside each encoder layer.
        dropout: Dropout probability.
        n_classes: Number of output intention classes.
        max_time_steps: Maximum sequence length supported.
        predict_future_steps: If ``> 0``, attaches a future-prediction head
            (accessed via ``self.future_head``). The default ``forward`` call
            does not invoke it; use it separately when needed.

    Example::

        >>> model = CrossAttentionFusion(visual_dim=256, pose_dim=3,
        ...                              motion_dim=2, goal_dim=3,
        ...                              d_model=128, n_classes=2)
        >>> vis  = torch.randn(4, 30, 256)
        >>> pose = torch.randn(4, 30, 3)
        >>> goal = torch.randn(4, 30, 3)
        >>> logits = model(vis, head_pose=pose, goal=goal)
        >>> logits.shape
        torch.Size([4, 2])
    """

    def __init__(
        self,
        visual_dim: int,
        pose_dim: int = 3,
        gaze_dim: int = 0,
        motion_dim: int = 2,
        vel_dim: int = 0,
        goal_dim: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_classes: int = 2,
        max_time_steps: int = 64,
        predict_future_steps: int = 0,
    ) -> None:
        super().__init__()

        self.visual_dim = int(visual_dim)
        self.pose_dim = int(pose_dim)
        self.gaze_dim = int(gaze_dim)
        self.motion_dim = int(motion_dim)
        self.goal_dim = int(goal_dim)
        self.d_model = int(d_model)
        self.max_time_steps = int(max_time_steps)

        # Per-modality projections + LayerNorms
        self.visual_proj = nn.Linear(self.visual_dim, d_model) if self.visual_dim > 0 else None
        self.pose_proj   = nn.Linear(self.pose_dim,   d_model) if self.pose_dim   > 0 else None
        self.gaze_proj   = nn.Linear(self.gaze_dim,   d_model) if self.gaze_dim   > 0 else None
        self.motion_proj = nn.Linear(self.motion_dim, d_model) if self.motion_dim > 0 else None
        self.goal_proj   = nn.Linear(self.goal_dim,   d_model) if self.goal_dim   > 0 else None

        self.visual_ln = nn.LayerNorm(d_model) if self.visual_proj is not None else None
        self.pose_ln   = nn.LayerNorm(d_model) if self.pose_proj   is not None else None
        self.gaze_ln   = nn.LayerNorm(d_model) if self.gaze_proj   is not None else None
        self.motion_ln = nn.LayerNorm(d_model) if self.motion_proj is not None else None
        self.goal_ln   = nn.LayerNorm(d_model) if self.goal_proj   is not None else None

        # Learned per-modality bias embeddings
        self.modality_emb = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(d_model))
            for name, proj in [
                ("vis", self.visual_proj), ("pose", self.pose_proj),
                ("gaze", self.gaze_proj), ("motion", self.motion_proj),
                ("goal", self.goal_proj),
            ]
            if proj is not None
        })

        # Temporal position embeddings
        self.time_emb = nn.Parameter(torch.randn(self.max_time_steps, d_model) * 0.02)

        # Self-attention over visual stream
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention: visual queries → auxiliary modality keys/values
        # Uses (L, N, E) convention so we transpose before calling
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.cross_attn_ln = nn.LayerNorm(d_model)

        # Adaptive goal gate: per-timestep sigmoid gate ∈ (0, 1)
        gate_hidden = max(8, d_model // 4)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        # CLS token + classifier
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        # Optional future-prediction head (not called by default forward)
        self.predict_future = int(predict_future_steps) > 0
        if self.predict_future:
            self.future_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, int(predict_future_steps) * 2),
            )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise projections with Xavier uniform; embeddings with N(0, 0.02)."""
        for name in ("visual_proj", "pose_proj", "gaze_proj", "motion_proj", "goal_proj"):
            proj: Optional[nn.Linear] = getattr(self, name)
            if proj is not None:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        for p in self.modality_emb.values():
            nn.init.normal_(p, std=0.02)

    def _project(
        self,
        tensor: torch.Tensor,
        proj: Optional[nn.Linear],
        ln: Optional[nn.LayerNorm],
        emb_key: str,
    ) -> torch.Tensor:
        """Project a modality tensor, apply LayerNorm, and add its bias embedding."""
        out = proj(tensor) if proj is not None else tensor
        if ln is not None:
            out = ln(out)
        if emb_key in self.modality_emb:
            out = out + self.modality_emb[emb_key]
        return out

    def forward(
        self,
        visual_feats: torch.Tensor,
        head_pose: Optional[torch.Tensor] = None,
        motion: Optional[torch.Tensor] = None,
        gaze: Optional[torch.Tensor] = None,
        vel: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            visual_feats: ``(B, T, visual_dim)`` visual features — required.
            head_pose: ``(B, T, pose_dim)`` head-pose features or ``None``.
            motion: ``(B, T, motion_dim)`` motion features or ``None``.
            gaze: ``(B, T, gaze_dim)`` gaze features or ``None``.
            vel: Accepted for API consistency; ignored internally.
            goal: ``(B, T, goal_dim)`` goal representation or ``None``.

        Returns:
            logits: ``(B, n_classes)``
        """
        B, T, _ = visual_feats.shape

        # 1) Project visual and self-attend
        v_emb = self._project(visual_feats, self.visual_proj, self.visual_ln, "vis")
        visual_ctx = self.transformer(v_emb)   # (B, T, d)

        # 2) Cross-attend from visual queries to each auxiliary modality
        aux_sources: list[torch.Tensor] = []
        if head_pose is not None and self.pose_proj is not None:
            aux_sources.append(self._project(head_pose, self.pose_proj, self.pose_ln, "pose"))
        if motion is not None and self.motion_proj is not None:
            aux_sources.append(self._project(motion, self.motion_proj, self.motion_ln, "motion"))
        if gaze is not None and self.gaze_proj is not None:
            aux_sources.append(self._project(gaze, self.gaze_proj, self.gaze_ln, "gaze"))

        fused = visual_ctx
        if aux_sources:
            q = fused.transpose(0, 1)           # (T, B, d)
            attn_sum: Optional[torch.Tensor] = None
            for src in aux_sources:
                kv = src.transpose(0, 1)        # (T, B, d)
                out, _ = self.cross_attn(q, kv, kv)
                attn_sum = out if attn_sum is None else attn_sum + out
            fused = fused + self.cross_attn_ln(attn_sum.transpose(0, 1))

        # 3) Add temporal position embeddings
        fused = fused + self.time_emb[:T].unsqueeze(0).to(fused.device)

        # 4) Adaptive goal conditioning
        if goal is not None and self.goal_proj is not None:
            g_emb = self._project(goal, self.goal_proj, self.goal_ln, "goal")
            gate = self.gate_net(fused)         # (B, T, 1)
            fused = (1.0 - gate) * fused + gate * g_emb

        # 5) Second self-attention pass, then pool and classify
        x = self.transformer(fused)
        pooled = x.mean(dim=1) + self.cls_token.squeeze(1)
        return self.classifier(pooled)


if __name__ == "__main__":
    B, T = 2, 30
    model = CrossAttentionFusion(
        visual_dim=256, pose_dim=3, gaze_dim=2, motion_dim=2, goal_dim=3,
        d_model=128, n_classes=2,
    )
    vis  = torch.randn(B, T, 256)
    pose = torch.randn(B, T, 3)
    mot  = torch.randn(B, T, 2)
    gaze = torch.randn(B, T, 2)
    goal = torch.randn(B, T, 3)
    logits = model(vis, head_pose=pose, motion=mot, gaze=gaze, goal=goal)
    print("logits.shape:", logits.shape)   # (2, 2)
