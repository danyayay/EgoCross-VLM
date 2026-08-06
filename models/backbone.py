"""Visual backbone architectures for feature extraction.

This module provides various backbone architectures for extracting visual
features from video frames. Supported backbones include simple CNNs and
pre-trained models like CLIP.
"""

from typing import Optional

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
	"""Small CNN to extract per-frame embeddings from raw frames.

	A lightweight convolutional neural network that processes individual frames
	and produces embeddings suitable for downstream tasks. The architecture uses
	depthwise convolutions followed by global average pooling and fully connected
	layers to produce fixed-dimension embeddings.

	Args:
		in_ch (int): Number of input channels. Default: 3 (RGB).
		out_dim (int): Dimension of output embeddings. Default: 256.
		hidden (int): Number of hidden channels in first conv layer. Default: 64.

	Input:
		frames (torch.Tensor): Batch of video frames with shape (B, T, H, W, 3),
			where B is batch size, T is number of frames, H and W are height and
			width, and values should be in range [0, 1].

	Output:
		torch.Tensor: Per-frame embeddings with shape (B, T, out_dim).

	Example:
		>>> backbone = CNNBackbone(in_ch=3, out_dim=256)
		>>> frames = torch.rand(4, 8, 224, 224, 3)  # 4 videos, 8 frames each
		>>> embeddings = backbone(frames)
		>>> embeddings.shape
		torch.Size([4, 8, 256])
	"""

	def __init__(self, in_ch: int = 3, out_dim: int = 256, hidden: int = 64):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, hidden, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(hidden),
			nn.GELU(),
			nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(hidden * 2),
			nn.GELU(),
			nn.Conv2d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(hidden * 4),
			nn.GELU(),
		)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(nn.Linear(hidden * 4, out_dim), nn.GELU())

	def forward(self, frames: torch.Tensor) -> torch.Tensor:
		"""Extract embeddings from video frames.
		
		Args:
			frames (torch.Tensor): Input frames with shape (B, T, H, W, 3).
			
		Returns:
			torch.Tensor: Output embeddings with shape (B, T, out_dim).
		"""
		# frames: (B, T, H, W, 3) -> process as (B*T, C, H, W)
		B, T, H, W, C = frames.shape
		x = frames.permute(0, 1, 4, 2, 3).contiguous()  # B,T,C,H,W
		x = x.view(B * T, C, H, W)
		x = self.conv(x)
		x = self.pool(x).view(B * T, -1)
		x = self.fc(x)
		x = x.view(B, T, -1)
		return x


class CLIPBackbone(nn.Module):
	"""Wrapper around OpenAI CLIP visual encoder for feature extraction.

	Uses pre-trained CLIP model to extract per-frame visual embeddings.
	Supports freezing parameters for use as feature extractor without fine-tuning.

	Args:
		model_name (str): CLIP model identifier (e.g., "ViT-B/32", "ViT-L/14").
			Default: "ViT-B/32".
		device (torch.device, optional): Device to place model on. If None,
			uses CPU. Default: None.
		freeze (bool): Whether to freeze model parameters. Default: False.

	Attributes:
		out_dim (int): Dimension of output embeddings.

	Raises:
		ImportError: If CLIP package is not installed.

	Example:
		>>> backbone = CLIPBackbone(model_name="ViT-B/32", freeze=True)
		>>> frames = torch.rand(4, 8, 224, 224, 3)
		>>> embeddings = backbone(frames)
		>>> embeddings.shape
		torch.Size([4, 8, 512])

	Note:
		- Requires CLIP to be installed: `pip install git+https://github.com/openai/CLIP.git`
		- Input frames should be normalized to [0, 1] range
		- CLIP internally applies standard ImageNet normalization
	"""

	def __init__(self, model_name: str = "ViT-B/32", device: Optional[torch.device] = None, freeze: bool = False):
		super().__init__()
		try:
			import clip
		except Exception as e:
			raise ImportError("clip package is required for CLIPBackbone. Install via `pip install git+https://github.com/openai/CLIP.git` or provide a different encoder.") from e

		dev = device or torch.device("cpu")
		self.model, _ = clip.load(model_name, device=dev, jit=False)
		self.model = self.model.float()
		# set train/eval according to freeze
		if freeze:
			for p in self.model.parameters():
				p.requires_grad = False

	@property
	def out_dim(self) -> int:
		"""Get output dimension of CLIP embeddings.
		
		Returns:
			int: Embedding dimension (typically 512 for ViT-B/32).
		"""
		# CLIP visual dimensionality
		if hasattr(self.model.visual, "output_dim"):
			return self.model.visual.output_dim
		if hasattr(self.model, "output_dim"):
			return self.model.output_dim
		# fallback
		return 512

	def forward(self, frames: torch.Tensor) -> torch.Tensor:
		"""Extract CLIP embeddings from video frames.
		
		Args:
			frames (torch.Tensor): Input frames with shape (B, T, H, W, 3),
				values should be in [0, 1] range.
		
		Returns:
			torch.Tensor: CLIP embeddings with shape (B, T, out_dim).
			
		Note:
			Applies CLIP's standard ImageNet normalization internally.
		"""
		# frames: (B, T, H, W, 3)
		B, T, H, W, C = frames.shape
		x = frames.permute(0, 1, 4, 2, 3).contiguous().view(B * T, C, H, W)
		x = x.to(next(self.model.parameters()).device)
		# normalize with CLIP stats
		mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
		std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
		x = (x - mean) / std

		feats = self.model.encode_image(x)
		feats = feats.view(B, T, -1)
		return feats
