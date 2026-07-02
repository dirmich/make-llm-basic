"""임베딩 레이어 — 토큰 ID를 밀집 벡터로 변환.

두 가지 임베딩을 다룸:
  1. TokenEmbedding: 토큰 ID → 밀집 벡터 (학습 가능한 룩업 테이블)
  2. PositionalEmbedding: 위치 정보 주입 (sinusoidal 또는 학습 가능)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """토큰 ID를 밀집 벡터로 변환하는 룩업 테이블.

    nn.Embedding의 얇은 래퍼. 가중치 초기화를 정규분포(N(0, d_model^-0.5))로 수행하여
    초기 임베딩 값이 너무 크거나 작지 않도록 함 (GPT-2 스타일).
    """

    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self._init_weights()

    def _init_weights(self) -> None:
        """정규분포 초기화. 표준편차는 1/sqrt(d_model)."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        # 패딩 토큰은 0으로
        with torch.no_grad():
            self.embedding.weight[self.pad_id].fill_(0.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [batch, seq] → [batch, seq, d_model]"""
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """위치 임베딩. 두 가지 모드 지원:

    - "sinusoidal": sin/cos 고정 위치 인코딩 (Vaswani et al. 2017)
    - "learned": 학습 가능한 위치 임베딩 (GPT-2 스타일)
    """

    def __init__(self, d_model: int, max_len: int = 512, mode: str = "learned"):
        super().__init__()
        assert mode in ("sinusoidal", "learned"), f"Unknown mode: {mode}"
        self.d_model = d_model
        self.max_len = max_len
        self.mode = mode

        if mode == "sinusoidal":
            # 고정 sin/cos 위치 인코딩
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # [1, max_len, d_model]로 저장
            self.register_buffer("pe", pe.unsqueeze(0))
        else:
            # 학습 가능한 위치 임베딩
            self.pe = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.pe.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

    def forward(self, seq_len: int) -> torch.Tensor:
        """길이 seq_len의 위치 임베딩 반환. shape: [1, seq_len, d_model]"""
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}. "
                f"Instantiate with larger max_len."
            )
        if self.mode == "sinusoidal":
            return self.pe[:, :seq_len, :]
        else:
            positions = torch.arange(seq_len, device=self.pe.weight.device)
            return self.pe(positions).unsqueeze(0)


class EmbeddingStack(nn.Module):
    """Token + Position 임베딩을 합친 모듈. 드롭아웃 포함."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        pad_id: int = 0,
        pos_mode: str = "learned",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)
        self.pos_emb = PositionalEmbedding(d_model, max_len, mode=pos_mode)
        self.dropout = nn.Dropout(dropout)
        # 스케일링: 임베딩에 sqrt(d_model)을 곱함 (Vaswani et al. 제안)
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [batch, seq] → [batch, seq, d_model]"""
        batch, seq = token_ids.shape
        tok = self.token_emb(token_ids) * self.scale
        pos = self.pos_emb(seq)
        return self.dropout(tok + pos)
