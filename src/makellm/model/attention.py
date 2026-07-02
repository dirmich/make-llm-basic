"""어텐션 메커니즘 — 트랜스포머의 핵심.

세 가지 컴포넌트를 다룸:
  1. ScaledDotProductAttention: Q·K^T/√d → softmax → V
  2. CausalMask: 미래 토큰을 가리는 마스크
  3. MultiHeadAttention: 어텐션을 헤드별로 분할하여 병렬 수행
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from ..utils.config import ModelConfig


def make_causal_mask(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """하삼각 마스크 생성. 미래 토큰을 -inf로 마스킹.

    Returns:
        mask: [seq_len, seq_len]의 부동소수 텐서
              마스킹 위치는 -inf, 통과 위치는 0.0
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # 주대각선 위를 -inf로
    return mask


class CausalMask:
    """인과적 마스크 (causal mask) 유틸리티 클래스.

    단순히 함수 래퍼이지만 개념을 명확히 하기 위해 클래스로 캡슐화.
    """

    @staticmethod
    def create(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return make_causal_mask(seq_len, device)

    @staticmethod
    def apply(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """어텐션 점수에 마스크를 적용. scores + mask."""
        return scores + mask


class ScaledDotProductAttention(nn.Module):
    """Q·K^T/√d_head → softmax → V.

    단일 헤드 어텐션의 가장 기본 형태.
    PyTorch 2.0의 F.scaled_dot_product_attention(SDPA)를 활용하여
    메모리 효율적이고 빠른 구현을 제공.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout_p = dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q, k, v: [batch, n_heads, seq, d_head]
            mask: [batch, 1, seq, seq] 또는 [seq, seq]의 어텐션 마스크
                  마스킹 위치는 -inf, 통과 위치는 0.0
        Returns:
            output: [batch, n_heads, seq, d_head]
        """
        # PyTorch 2.0 SDPA 사용
        # - is_causal=True를 쓰면 내부적으로 인과 마스크를 적용하지만
        #   외부에서 패딩 마스크 등과 결합하기 어려워 명시적 마스크를 선호
        attn_mask = mask  # 마스크는 boolean 또는 -inf/0 부동소수
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return out


class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션 (Vaswani et al. 2017).

    d_model을 n_heads로 분할하여 각 헤드가 d_model/n_heads 차원에서
    독립적으로 어텐션을 수행한 뒤 다시 concat.

    구성:
      - W_Q, W_K, W_V: [d_model, d_model] 선형 변환
      - W_O: [d_model, d_model] 출력 변환
      - 어텐션 스케일: 1/sqrt(d_head)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head  # = d_model // n_heads
        assert self.d_model == self.n_heads * self.d_head, (
            f"d_model({self.d_model}) != n_heads({self.n_heads}) * d_head({self.d_head})"
        )

        # Q, K, V를 한 번에 계산하기 위한 결합 프로젝션 (효율성)
        # 입력 [batch, seq, d_model] → [batch, seq, 3*d_model]
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=config.use_bias)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)

        self.attention = ScaledDotProductAttention(dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        # 선형 레이어 초기화 (GPT-2 스타일)
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, d_model]
            mask: [seq, seq] 또는 [batch, 1, seq, seq]
        Returns:
            output: [batch, seq, d_model]
        """
        batch, seq, _ = x.shape

        # Q, K, V 계산: [batch, seq, 3*d_model]
        qkv = self.qkv_proj(x)
        # 분할: [batch, seq, 3, n_heads, d_head]
        qkv = qkv.view(batch, seq, 3, self.n_heads, self.d_head)
        # 전치: [3, batch, n_heads, seq, d_head]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 마스크 준비: [seq, seq] → [1, 1, seq, seq]로 브로드캐스트
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # 어텐션: [batch, n_heads, seq, d_head]
        attn_out = self.attention(q, k, v, mask=mask)

        # 헤드 합치기: [batch, seq, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

        # 출력 프로젝션
        return self.dropout(self.o_proj(attn_out))
