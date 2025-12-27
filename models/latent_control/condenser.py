from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CondenserConfig:
    # 输出固定 M 个 memory tokens（避免序列长度不可控）
    memory_tokens: int = 8
    # 注意力头数（需要整除 d_model）
    num_heads: int = 8
    # 是否对输出做一层 MLP
    mlp_ratio: float = 2.0


class AttentionCondenser(nn.Module):
    """
    Condenser: 从“图像 token hidden states 序列”提炼出固定长度的 memory tokens。

    输入:
      - h_img_seq: [B, S, D]  (S 可以是窗口内最近若干步的 image-token hidden states)

    输出:
      - m_tokens: [B, M, D]  (固定 M 个 memory tokens，处于 language model 的 embedding space)
      - m_vec:    [B, D]     (对 m_tokens 做 mean pooling 的向量表示，便于 Trigger/Translator 使用)
    """

    def __init__(self, d_model: int, cfg: CondenserConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model

        self.latents = nn.Parameter(torch.randn(cfg.memory_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=cfg.num_heads, batch_first=True
        )
        hidden = int(d_model * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, h_img_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if h_img_seq.dim() != 3:
            raise ValueError(f"h_img_seq must be [B,S,D], got {tuple(h_img_seq.shape)}")
        b, _, d = h_img_seq.shape
        if d != self.d_model:
            raise ValueError(f"d_model mismatch: expected {self.d_model}, got {d}")

        # [B, M, D]
        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        # Cross-attn: latents attend to image-token sequence
        m_tokens, _ = self.cross_attn(query=q, key=h_img_seq, value=h_img_seq, need_weights=False)
        m_tokens = m_tokens + self.mlp(m_tokens)
        m_vec = m_tokens.mean(dim=1)
        return m_tokens, m_vec


class MeanMLPCondenser(nn.Module):
    """
    更简单的 Condenser：mean pooling + MLP，再复制成 M 个 tokens。
    用于快速跑通/对齐 shape。
    """

    def __init__(self, d_model: int, cfg: CondenserConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        hidden = int(d_model * cfg.mlp_ratio)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, h_img_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if h_img_seq.dim() != 3:
            raise ValueError(f"h_img_seq must be [B,S,D], got {tuple(h_img_seq.shape)}")
        pooled = h_img_seq.mean(dim=1)
        m_vec = self.proj(pooled)
        m_tokens = m_vec.unsqueeze(1).expand(-1, self.cfg.memory_tokens, -1).contiguous()
        return m_tokens, m_vec


