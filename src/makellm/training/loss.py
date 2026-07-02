"""손실 함수 — 언어 모델링용 cross-entropy와 label smoothing 변종."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """표준 cross-entropy 손실.

    Args:
        logits: [batch, seq, vocab_size]
        targets: [batch, seq] (long)
        ignore_index: 이 인덱스는 손실에서 제외 (예: 패딩)
    Returns:
        scalar loss
    """
    batch, seq, vocab = logits.shape
    # F.cross_entropy는 [N, C] / [N] 형태를 기대
    logits_flat = logits.view(-1, vocab)
    targets_flat = targets.view(-1)
    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


def label_smoothing_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Label smoothing이 적용된 cross-entropy.

    정답 토큰에 1-smoothing 확률을 주고, 나머지 토큰에 smoothing/vocab 확률을
    균등하게 분배. 과적합 방지와 일반화 성능 향상에 도움.

    Args:
        logits: [batch, seq, vocab_size]
        targets: [batch, seq]
        smoothing: 0.0~1.0 (0이면 일반 CE와 동일)
        ignore_index: 손실에서 제외할 인덱스
    """
    batch, seq, vocab = logits.shape
    logits_flat = logits.view(-1, vocab)
    targets_flat = targets.view(-1)

    # log-softmax
    log_probs = F.log_softmax(logits_flat, dim=-1)
    # uniform 분포
    nll_loss = -log_probs.gather(1, targets_flat.clamp(min=0).unsqueeze(1)).squeeze(1)
    smooth_loss = -log_probs.mean(dim=-1)

    # ignore_index 마스크
    mask = (targets_flat != ignore_index).float()
    nll_loss = (nll_loss * mask).sum() / mask.sum().clamp(min=1)
    smooth_loss = (smooth_loss * mask).sum() / mask.sum().clamp(min=1)

    return (1.0 - smoothing) * nll_loss + smoothing * smooth_loss


def compute_perplexity(loss: float) -> float:
    """손실값으로부터 perplexity 계산. PPL = exp(loss)."""
    import math
    return math.exp(min(loss, 20.0))  # overflow 방지
