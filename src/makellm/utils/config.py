"""설정 데이터클래스 — 모든 하이퍼파라미터를 한 곳에서 관리."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json


@dataclass
class TokenizerConfig:
    """토크나이저 학습 설정."""

    algorithm: str = "bpe"          # "bpe" 또는 "unigram"
    vocab_size: int = 1000          # 목표 어휘 크기
    special_tokens: list[str] = field(
        default_factory=lambda: ["<pad>", "<bos>", "<eos>", "<unk>"]
    )
    character_coverage: float = 1.0  # Unigram에서 사용 (0.0~1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenizerConfig":
        return cls(**d)


@dataclass
class ModelConfig:
    """GPT 모델 아키텍처 설정."""

    vocab_size: int = 1000
    context_length: int = 128        # 최대 시퀀스 길이
    d_model: int = 128               # 임베딩 차원
    n_heads: int = 4                 # 어텐션 헤드 수
    n_layers: int = 4                # 트랜스포머 블록 수
    d_ff: int = 512                  # FFN 은닉 차원
    dropout: float = 0.1             # 드롭아웃 비율
    use_bias: bool = True            # Linear 바이어스 사용 여부
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**d)

    @property
    def d_head(self) -> int:
        """헤드당 차원. d_model이 n_heads로 나누어떨어져야 함."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model({self.d_model}) must be divisible by n_heads({self.n_heads})"
        )
        return self.d_model // self.n_heads


@dataclass
class TrainerConfig:
    """학습 루프 설정."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 10
    max_steps: int | None = None     # None이면 max_epochs 사용
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"     # "cosine", "linear", "constant"
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 200
    save_every: int = 1000
    seed: int = 42
    device: str = "cpu"              # "cpu" 또는 "cuda"
    out_dir: str = "./runs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainerConfig":
        return cls(**d)


def save_config(cfg: Any, path: str | Path) -> None:
    """설정을 JSON 파일로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)


def load_config(cls_type: type, path: str | Path):
    """JSON 파일에서 설정 로드. cls_type은 TokenizerConfig/ModelConfig/TrainerConfig 중 하나."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return cls_type.from_dict(d)
