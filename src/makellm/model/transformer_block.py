"""트랜스포머 블록 — 멀티헤드 어텐션 + FFN + 잔차 + LayerNorm.

두 가지 정규화 방식을 다룸:
  - "post-LN": 어텐션/FFN 다음에 LayerNorm (Vaswani 원본)
  - "pre-LN": LayerNorm 다음에 어텐션/FFN (GPT-2 스타일, 학습 안정성 우수)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import ModelConfig
from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """포지션 와이즈 FFN. 2개의 선형 레이어 + 활성화 함수.

    구조: Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)
    d_ff는 보통 d_model의 4배.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq, d_model] → [batch, seq, d_model]"""
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    """단일 트랜스포머 디코더 블록.

    구조 (pre-LN):
        x → LN1 → MHA → + residual → LN2 → FFN → + residual → out

    pre-LN이 학습 안정성이 좋아 현대 LLM에서 표준으로 사용됨.
    """

    def __init__(self, config: ModelConfig, layer_norm_style: str = "pre"):
        super().__init__()
        self.config = config
        self.layer_norm_style = layer_norm_style
        assert layer_norm_style in ("pre", "post"), f"Unknown style: {layer_norm_style}"

        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [batch, seq, d_model] → [batch, seq, d_model]"""
        if self.layer_norm_style == "pre":
            # Pre-LN: 정규화 → 어텐션 → 잔차
            x = x + self.attn(self.ln1(x), mask=mask)
            x = x + self.ffn(self.ln2(x))
        else:
            # Post-LN: 어텐션 → 잔차 → 정규화
            x = self.ln1(x + self.attn(x, mask=mask))
            x = self.ln2(x + self.ffn(x))
        return x
