"""옵티마이저와 학습률 스케줄러 빌더."""

from __future__ import annotations

import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..utils.config import TrainerConfig


def build_optimizer(model: torch.nn.Module, config: TrainerConfig) -> AdamW:
    """AdamW 옵티마이저 생성.

    가중치 감쇠(weight decay)는 바이어스와 LayerNorm에는 적용하지 않는 것이
    관행 (과적합 방지 효과가 없고 학습을 불안정하게 만들 수 있음).
    """
    # 파라미터 그룹 분리: decay / no_decay
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "layernorm" in name.lower() or "ln" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
    total_steps: int,
) -> LambdaLR:
    """학습률 스케줄러 생성.

    세 가지 모드:
      - "cosine": warmup 후 코사인 감쇠 (GPT-3 스타일)
      - "linear": warmup 후 선형 감쇠
      - "constant": warmup 후 일정
    """
    warmup = config.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            # 선형 warmup
            return float(step) / float(max(1, warmup))
        # warmup 이후
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(progress, 1.0)
        if config.lr_scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif config.lr_scheduler == "linear":
            return max(0.0, 1.0 - progress)
        elif config.lr_scheduler == "constant":
            return 1.0
        else:
            raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")

    return LambdaLR(optimizer, lr_lambda)
