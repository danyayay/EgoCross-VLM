"""LSTM-based multimodal fusion model.

Projects the visual stream to a shared embedding dimension and concatenates
all available modality features along the feature axis before feeding the
resulting sequence into a unidirectional LSTM.

Setting ``visual_dim=0`` produces a non-visual baseline that operates purely
on scalar modalities (head pose, gaze, motion, velocity, goal).
"""

from typing import Optional

import torch
import torch.nn as nn


class LSTMFusion(nn.Module):
    """LSTM fusion model for multimodal egocentric time-series.

    Concatenates all available modalities along the feature dimension at each
    timestep and processes the resulting sequence through a stacked
    unidirectional LSTM. The last hidden state is projected to class logits.

    The visual stream is first projected from ``visual_dim`` to ``d_model``
    with a linear layer and LayerNorm before concatenation. All other
    modalities (pose, gaze, motion, velocity, goal) are concatenated directly
    without projection.

    Args:
        visual_dim: Dimensionality of visual features per timestep.
            Set to ``0`` to disable the visual modality.
        pose_dim: Dimensionality of head-pose features (0 to disable).
        gaze_dim: Dimensionality of gaze features (0 to disable).
        motion_dim: Dimensionality of body-frame motion features (0 to disable).
        vel_dim: Dimensionality of velocity features (0 to disable).
        goal_dim: Dimensionality of goal features (0 to disable).
        d_model: Output dimension of the visual projection layer. Ignored when
            ``visual_dim=0``.
        hidden_dim: Number of hidden units in the LSTM.
        num_layers: Number of stacked LSTM layers.
        n_classes: Number of output intention classes.

    Example::

        >>> model = LSTMFusion(visual_dim=256, motion_dim=2, vel_dim=2,
        ...                    d_model=128, hidden_dim=64, n_classes=2)
        >>> vis = torch.randn(8, 30, 256)   # (B, T, visual_dim)
        >>> mot = torch.randn(8, 30, 2)
        >>> vel = torch.randn(8, 30, 2)
        >>> logits = model(vis, motion=mot, vel=vel)
        >>> logits.shape
        torch.Size([8, 2])
    """

    def __init__(
        self,
        visual_dim: int = 0,
        pose_dim: int = 0,
        gaze_dim: int = 0,
        motion_dim: int = 0,
        vel_dim: int = 0,
        goal_dim: int = 0,
        d_model: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        n_classes: int = 2,
    ) -> None:
        super().__init__()

        self.visual_dim = visual_dim
        self.pose_dim = pose_dim
        self.gaze_dim = gaze_dim
        self.motion_dim = motion_dim
        self.vel_dim = vel_dim
        self.goal_dim = goal_dim
        self.d_model = d_model

        # Visual projection (only when a visual stream is present)
        self.visual_proj = nn.Linear(visual_dim, d_model) if visual_dim > 0 else None
        self.visual_ln = nn.LayerNorm(d_model) if visual_dim > 0 else None

        # LSTM input = projected visual dim (or 0) + raw scalar modalities
        lstm_input_dim = (d_model if visual_dim > 0 else 0) + pose_dim + gaze_dim + motion_dim + vel_dim + goal_dim
        if lstm_input_dim == 0:
            raise ValueError("At least one modality must have a non-zero dimension.")

        self.rnn = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, n_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise weights with Xavier uniform / normal initialisation."""
        if self.visual_proj is not None:
            nn.init.xavier_normal_(self.visual_proj.weight)
            nn.init.zeros_(self.visual_proj.bias)
        if self.visual_ln is not None:
            nn.init.ones_(self.visual_ln.weight)
            nn.init.zeros_(self.visual_ln.bias)
        for m in self.rnn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        visual_feats: Optional[torch.Tensor] = None,
        head_pose: Optional[torch.Tensor] = None,
        gaze: Optional[torch.Tensor] = None,
        motion: Optional[torch.Tensor] = None,
        vel: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            visual_feats: ``(B, T, visual_dim)`` visual features or ``None``.
            head_pose: ``(B, T, pose_dim)`` head-pose features or ``None``.
            gaze: ``(B, T, gaze_dim)`` gaze features or ``None``.
            motion: ``(B, T, motion_dim)`` motion features or ``None``.
            vel: ``(B, T, vel_dim)`` velocity features or ``None``.
            goal: ``(B, T, goal_dim)`` goal features or ``None``.

        Returns:
            logits: ``(B, n_classes)``
        """
        parts: list[torch.Tensor] = []

        if self.visual_proj is not None and visual_feats is not None:
            v = self.visual_proj(visual_feats)
            if self.visual_ln is not None:
                v = self.visual_ln(v)
            parts.append(v)

        if self.pose_dim > 0 and head_pose is not None:
            parts.append(head_pose)
        if self.gaze_dim > 0 and gaze is not None:
            parts.append(gaze)
        if self.motion_dim > 0 and motion is not None:
            parts.append(motion)
        if self.vel_dim > 0 and vel is not None:
            parts.append(vel)
        if self.goal_dim > 0 and goal is not None:
            parts.append(goal)

        x = torch.cat(parts, dim=-1)          # (B, T, input_dim)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])          # (B, n_classes)


if __name__ == "__main__":
    # quick shape check
    B, T = 4, 30
    model = LSTMFusion(visual_dim=256, motion_dim=2, vel_dim=2, d_model=128, hidden_dim=64, n_classes=2)
    vis = torch.randn(B, T, 256)
    mot = torch.randn(B, T, 2)
    vel = torch.randn(B, T, 2)
    logits = model(vis, motion=mot, vel=vel)
    print("logits.shape:", logits.shape)   # (4, 2)

    # motion-only baseline (no visual)
    model_nv = LSTMFusion(motion_dim=2, vel_dim=2, hidden_dim=64, n_classes=2)
    logits_nv = model_nv(motion=mot, vel=vel)
    print("logits_nv.shape:", logits_nv.shape)  # (4, 2)
