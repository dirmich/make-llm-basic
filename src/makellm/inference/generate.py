"""텍스트 생성 — 자기회귀 디코딩.

모델이 토큰을 한 번에 하나씩 생성하여 시퀀스를 만들어냄.
각 스텝에서 이전까지 생성된 토큰을 모델에 넣고 다음 토큰의 분포를 얻음.
"""

from __future__ import annotations

import torch

from ..model.gpt import GPT
from ..tokenizer.base import Tokenizer
from .sampler import GreedySampler


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    sampler=None,
    device: str | torch.device = "cpu",
    stop_on_eos: bool = True,
) -> str:
    """텍스트 생성 (자기회귀 디코딩).

    Args:
        model: 학습된 GPT 모델
        tokenizer: 토크나이저
        prompt: 프롬프트 문자열
        max_new_tokens: 생성할 최대 토큰 수
        sampler: 샘플러 (None이면 greedy)
        device: 'cpu' 또는 'cuda'
        stop_on_eos: <eos> 토큰이 나오면 중지
    Returns:
        생성된 텍스트 (프롬프트 포함)
    """
    if sampler is None:
        sampler = GreedySampler()

    model.eval()
    model.to(device)
    # 프롬프트 인코딩 (<bos> 추가)
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    eos_id = tokenizer.special.eos_id
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        # 컨텍스트 길이 제한 (최대 context_length까지만 사용)
        if input_ids.shape[1] > context_length:
            x = input_ids[:, -context_length:]
        else:
            x = input_ids
        # 순전파: 마지막 위치의 로짓만 사용
        logits = model(x)
        next_logits = logits[:, -1, :]  # [1, vocab_size]
        # 샘플링
        next_id = sampler(next_logits)  # [1]
        # 시퀀스에 추가
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
        # 종료 조건
        if stop_on_eos and next_id.item() == eos_id:
            break

    # 디코딩
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text


@torch.no_grad()
def generate_batch(
    model: GPT,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_new_tokens: int = 50,
    sampler=None,
    device: str | torch.device = "cpu",
) -> list[str]:
    """배치 단위 생성. 여러 프롬프트를 동시에 처리."""
    if sampler is None:
        sampler = GreedySampler()

    model.eval()
    model.to(device)
    bos_id = tokenizer.special.bos_id
    pad_id = tokenizer.special.pad_id

    # 각 프롬프트 인코딩
    batch_ids = [tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts]
    max_len = max(len(ids) for ids in batch_ids)
    # 패딩하여 배치 생성 (left-padding으로 생성 품질 유지)
    padded = []
    for ids in batch_ids:
        pad_len = max_len - len(ids)
        padded.append([pad_id] * pad_len + ids)
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)

    context_length = model.config.context_length
    eos_id = tokenizer.special.eos_id

    for _ in range(max_new_tokens):
        if input_ids.shape[1] > context_length:
            x = input_ids[:, -context_length:]
        else:
            x = input_ids
        logits = model(x)
        next_logits = logits[:, -1, :]
        next_ids = sampler(next_logits)  # [batch]
        input_ids = torch.cat([input_ids, next_ids.unsqueeze(1)], dim=1)

    # 디코딩
    results = []
    for row in input_ids:
        # pad_id 제거
        ids = [i for i in row.tolist() if i != pad_id]
        results.append(tokenizer.decode(ids))
    return results
