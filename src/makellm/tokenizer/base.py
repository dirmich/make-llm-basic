"""토크나이저 베이스 클래스 — 모든 토크나이저가 따라야 하는 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Sequence


@dataclass
class SpecialTokens:
    """특수 토큰 모음. 인덱스는 고정."""

    pad: str = "<pad>"      # 0
    bos: str = "<bos>"      # 1
    eos: str = "<eos>"      # 2
    unk: str = "<unk>"      # 3

    @property
    def tokens(self) -> list[str]:
        return [self.pad, self.bos, self.eos, self.unk]

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3


class Tokenizer(ABC):
    """모든 토크나이저의 추상 베이스.

    자식 클래스는 다음 메서드를 구현해야 함:
      - _train(corpus) : 어휘 학습
      - _tokenize(text) : 텍스트를 토큰 리스트로 분할
      - _token_to_id(token) / _id_to_token(token_id) : 매핑
    """

    def __init__(self, special_tokens: SpecialTokens | None = None):
        self.special = special_tokens or SpecialTokens()
        # id → token / token → id 양방향 매핑
        self.id_to_token: list[str] = list(self.special.tokens)
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.id_to_token)}
        self._trained = False

    # ============ 추상 메서드 ============

    @abstractmethod
    def _train(self, corpus: Sequence[str], vocab_size: int) -> None:
        """어휘 학습. self.id_to_token, self.token_to_id를 채움."""

    @abstractmethod
    def _tokenize(self, text: str) -> list[str]:
        """학습된 어휘로 텍스트를 토큰 리스트로 분할."""

    # ============ 공개 API ============

    def train(self, corpus: Sequence[str], vocab_size: int) -> None:
        """어휘 학습. 특수 토큰 4개를 제외한 만큼 학습."""
        target = max(vocab_size, len(self.special.tokens) + 1)
        self._train(corpus, target)
        self._trained = True

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """텍스트를 정수 토큰 ID 리스트로 변환.

        Args:
            text: 인코딩할 문자열
            add_bos: 시작 토큰 <bos> 추가 여부
            add_eos: 종료 토큰 <eos> 추가 여부
        Returns:
            토큰 ID 리스트
        """
        if not self._trained:
            raise RuntimeError("Tokenizer must be trained before encoding. Call .train() first.")
        tokens = self._tokenize(text)
        ids = [self._token_to_id(t) for t in tokens]
        if add_bos:
            ids = [self.special.bos_id] + ids
        if add_eos:
            ids = ids + [self.special.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """토큰 ID 리스트를 다시 문자열로 변환. 특수 토큰은 무시.

        BPE의 경우 토큰이 chr(byte) 형태일 수 있어, 각 토큰을 Latin-1로
        인코딩하여 원래 바이트를 복원한 뒤 전체를 UTF-8로 디코딩합니다.
        일반적인 멀티바이트 토큰(예: "▁the")은 자연스럽게 처리됩니다.
        """
        tokens: list[str] = []
        for i in ids:
            tok = self._id_to_token(int(i))
            if tok in self.special.tokens:
                continue
            tokens.append(tok)
        # 토큰을 바이트로 합침
        # - chr(byte) 토큰: Latin-1로 인코딩하여 1바이트 복원
        # - 일반 문자열 토큰: UTF-8로 인코딩
        # 두 방식을 혼합하면 안 되므로, BPE는 Latin-1, Unigram은 UTF-8 사용.
        # 여기서는 자식 클래스가 _decode_bytes를 오버라이드하도록 함.
        raw = self._tokens_to_bytes(tokens)
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = "".join(tokens)
        return self._postprocess(text)

    def _tokens_to_bytes(self, tokens: list[str]) -> bytes:
        """토큰 리스트를 바이트로 변환. 기본은 UTF-8 인코딩."""
        return b"".join(t.encode("utf-8") for t in tokens)

    def _token_to_id(self, token: str) -> int:
        """토큰을 ID로. 없으면 <unk>."""
        return self.token_to_id.get(token, self.special.unk_id)

    def _id_to_token(self, token_id: int) -> str:
        """ID를 토큰으로. 범위 밖이면 <unk>."""
        if 0 <= token_id < len(self.id_to_token):
            return self.id_to_token[token_id]
        return self.special.unk

    def _postprocess(self, text: str) -> str:
        """디코딩 후 처리. 자식 클래스에서 오버라이드 가능 (예: BPE의 공백 복원)."""
        return text

    # ============ 저장/로드 ============

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def save(self, path: str | Path) -> None:
        """토크나이저를 JSON 파일로 저장."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": self.__class__.__name__,
            "id_to_token": self.id_to_token,
            "special": {
                "pad": self.special.pad,
                "bos": self.special.bos,
                "eos": self.special.eos,
                "unk": self.special.unk,
            },
            "extra": self._save_extra(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_extra(self) -> dict:
        """자식 클래스에서 추가로 저장할 필드."""
        return {}

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        """JSON 파일에서 토크나이저 로드."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        special = SpecialTokens(**data["special"])
        inst = cls(special_tokens=special)  # type: ignore[arg-type]
        inst.id_to_token = data["id_to_token"]
        inst.token_to_id = {t: i for i, t in enumerate(inst.id_to_token)}
        inst._load_extra(data.get("extra", {}))
        inst._trained = True
        return inst

    def _load_extra(self, extra: dict) -> None:
        """자식 클래스에서 추가 필드 복원."""
        pass

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size}, trained={self._trained})"
