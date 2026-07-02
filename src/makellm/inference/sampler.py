"""샘플러 — 다음 토큰을 선택하는 전략들.

네 가지 전략을 구현:
  1. GreedySampler: 항상 가장 확률이 높은 토큰 선택 (결정론적)
  2. TemperatureSampler: 온도로 분포를 평평하게/날카롭게 만든 후 샘플링
  3. TopKSampler: 상위 K개 토큰만 고려하여 샘플링
  4. TopPSampler: 누적 확률이 P가 될 때까지만 고려 (nucleus sampling)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GreedySampler:
    """탐욕 샘플링 — 가장 높은 확률의 토큰 선택.

    항상 같은 입력에 같은 출력을 내므로 재현성이 중요할 때 사용.
    단조로운 텍스트를 생성하는 경향이 있음.
    """

    def __init__(self):
        pass

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [batch, vocab_size] → token_ids: [batch]"""
        return torch.argmax(logits, dim=-1)


class TemperatureSampler:
    """온도 샘플링.

    temperature < 1.0: 분포가 날카로워짐 (결정론적 경향)
    temperature > 1.0: 분포가 평평해짐 (다양성 증가)
    temperature = 1.0: 원래 분포 그대로
    """

    def __init__(self, temperature: float = 1.0):
        assert temperature > 0, "Temperature must be positive"
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [batch, vocab_size] → token_ids: [batch]"""
        probs = F.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TopKSampler:
    """Top-K 샘플링 — 상위 K개 토큰만 고려.

    분포의 꼬리 부분(확률이 매우 낮은 토큰)을 잘라내어
    비정상적인 토큰 생성을 방지.
    """

    def __init__(self, k: int = 50, temperature: float = 1.0):
        assert k > 0, "k must be positive"
        self.k = k
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [batch, vocab_size] → token_ids: [batch]"""
        scaled = logits / self.temperature
        # 상위 K개의 값과 인덱스
        top_values, top_indices = torch.topk(scaled, self.k, dim=-1)
        # 상위 K개에 대해서만 softmax
        probs = F.softmax(top_values, dim=-1)
        # 샘플링
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        # 원래 인덱스로 변환
        return top_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


class TopPSampler:
    """Top-P (Nucleus) 샘플링 — 누적 확률이 P가 될 때까지의 토큰만 고려.

    분포의 형태에 따라 고려하는 토큰 수가 동적으로 변함.
    분포가 날카로우면 적은 수의 토큰만, 평평하면 많은 토큰을 고려.
    """

    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        assert 0.0 < p <= 1.0, "p must be in (0, 1]"
        self.p = p
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [batch, vocab_size] → token_ids: [batch]"""
        scaled = logits / self.temperature
        probs = F.softmax(scaled, dim=-1)
        # 확률 내림차순 정렬
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        # 누적 확률
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # 누적이 p를 넘는 위치부터 마스킹
        # (p를 처음 넘는 위치는 포함시키기 위해 한 칸 shift)
        sorted_mask = cumulative - sorted_probs > self.p
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        # 정규화
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        # 샘플링
        sampled = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        return sorted_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
