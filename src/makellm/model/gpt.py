"""GPT 모델 — 트랜스포머 블록을 N층 쌓은 자기회귀 언어 모델.

구조:
    token_ids → [Token+Pos Embedding]
              → [TransformerBlock × N]
              → [Final LayerNorm]
              → [LM Head (d_model → vocab_size)]
              → logits
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from ..utils.config import ModelConfig
from .embedding import EmbeddingStack
from .transformer_block import TransformerBlock
from .attention import make_causal_mask


class GPT(nn.Module):
    """GPT 스타일 자기회귀 언어 모델."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = EmbeddingStack(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_len=config.context_length,
            pad_id=config.pad_token_id,
            pos_mode="learned",
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, layer_norm_style="pre") for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # LM Head: 임베딩 가중치를 공유(tied weights)하여 파라미터 절약
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # 가중치 타이: lm_head.weight = token_emb.embedding.weight
        self.lm_head.weight = self.embedding.token_emb.embedding.weight
        # 초기화는 이미 각 서브모듈에서 수행됨
        # 잔차 스케일링 (GPT-2 스타일): 출력 프로젝션을 1/sqrt(2*n_layers)로 스케일
        self._resid_scale = 1.0 / math.sqrt(2 * config.n_layers)
        self.apply(self._init_residuals)

    def _init_residuals(self, module: nn.Module) -> None:
        """잔차 스트림에 직접 기여하는 프로젝션 가중치를 작게 초기화.
        어텐션의 o_proj, FFN의 w2가 해당.
        """
        if isinstance(module, nn.Linear):
            if module.weight.shape[0] == self.config.d_model:
                # 출력 차원이 d_model인 경우 (잔차로 들어감)
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self,
        token_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """token_ids: [batch, seq] → logits: [batch, seq, vocab_size]"""
        batch, seq = token_ids.shape
        # 인과 마스크 생성
        mask = make_causal_mask(seq, device=token_ids.device)
        # 임베딩
        x = self.embedding(token_ids)
        # 트랜스포머 블록 통과
        for block in self.blocks:
            x = block(x, mask=mask)
        # 최종 정규화
        x = self.ln_f(x)
        # 로짓
        logits = self.lm_head(x)
        if return_hidden:
            return logits, x
        return logits

    @torch.no_grad()
    def num_parameters(self) -> int:
        """총 파라미터 수."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def num_trainable_parameters(self) -> int:
        """학습 가능한 파라미터 수."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        n = self.num_parameters()
        return (
            f"GPT(vocab={self.config.vocab_size}, d_model={self.config.d_model}, "
            f"n_layers={self.config.n_layers}, n_heads={self.config.n_heads}, "
            f"context={self.config.context_length}, params={n/1e6:.2f}M)"
        )
