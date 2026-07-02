"""언어 모델링용 데이터셋.

텍스트를 토큰 ID 시퀀스로 변환하고, 고정 길이 슬라이딩 윈도우로
(input, target) 쌍을 만들어냄. target은 input을 한 칸씩 shift한 것.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import Dataset

from ..tokenizer.base import Tokenizer


class LMDataset(Dataset):
    """언어 모델링용 데이터셋.

    텍스트를 토큰화한 뒤 context_length 크기의 청크로 분할.
    각 샘플은 (input_ids, target_ids) 형태이며 target은 input을 한 칸 shift.

    특징:
      - 슬라이딩 윈도우 (stride = context_length, 겹침 없음)
      - 마지막 청크가 짧으면 패딩
      - <bos>와 <eos>를 자동으로 추가
    """

    def __init__(
        self,
        text: str | Sequence[str],
        tokenizer: Tokenizer,
        context_length: int = 128,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.context_length = context_length
        self.tokenizer = tokenizer

        # 텍스트 결합
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        # 모든 텍스트를 토큰화하여 하나의 긴 시퀀스로
        all_ids: list[int] = []
        for t in texts:
            ids = tokenizer.encode(t, add_bos=add_bos, add_eos=add_eos)
            all_ids.extend(ids)
        self._ids = all_ids

        # 청크 분할
        self._chunks: list[list[int]] = []
        cl = context_length + 1  # input(context_length) + target shift 1
        for i in range(0, len(self._ids), cl):
            chunk = self._ids[i : i + cl]
            if len(chunk) < cl:
                # 마지막 청크 패딩
                pad_id = tokenizer.special.pad_id
                chunk = chunk + [pad_id] * (cl - len(chunk))
            self._chunks.append(chunk)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (input_ids, target_ids) 둘 다 [context_length]"""
        chunk = self._chunks[idx]
        # input: 첫 context_length개, target: 마지막 context_length개 (한 칸 shift)
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, target_ids

    @property
    def total_tokens(self) -> int:
        """전체 토큰 수."""
        return len(self._ids)


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """배치를 묶어서 [batch, seq] 텐서로 변환.

    모든 샘플이 같은 길이(context_length)이므로 단순 stack.
    """
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return inputs, targets


def load_text(path: str | Path) -> str:
    """텍스트 파일 로드."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_text_lines(path: str | Path) -> list[str]:
    """텍스트 파일을 줄 단위로 로드."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]
