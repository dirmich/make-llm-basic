"""BPE (Byte Pair Encoding) 토크나이저.

GPT-2, GPT-3, LLaMA 등 현대 LLM이 사용하는 토크나이저의 기반.
바이트 단위로 시작하여 가장 자주 등장하는 바이트 쌍을 병합해 나가는 방식.

참고: Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units"
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .base import Tokenizer


class BPETokenizer(Tokenizer):
    """바이트 단위 BPE 토크나이저.

    특징:
      - 모든 문자를 UTF-8 바이트로 표현 (unk 토큰이 거의 발생하지 않음)
      - 공백을 U+2581 (▁)로 치환하여 단어 경계 보존
      - 병합 규칙(merge rules)을 학습하여 저장
    """

    def __init__(self, special_tokens=None):
        super().__init__(special_tokens=special_tokens)
        self.merges: list[tuple[str, str]] = []   # 학습된 병합 규칙 (순서대로)
        self._merge_rank: dict[tuple[str, str], int] = {}

    # ---------- 학습 ----------

    def _train(self, corpus: Sequence[str], vocab_size: int) -> None:
        """코퍼스로부터 BPE 병합 규칙 학습.

        알고리즘:
          1. 코퍼스의 모든 텍스트를 바이트 단위 시퀀스로 변환
          2. 인접한 바이트 쌍의 빈도를 세어 가장 빈도가 높은 쌍을 병합
          3. 목표 어휘 크기에 도달할 때까지 반복
        """
        # 어휘 초기화: 특수 토큰 + 256개 바이트
        self.id_to_token = list(self.special.tokens)
        for b in range(256):
            self.id_to_token.append(self._byte_to_token(b))
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

        # 코퍼스를 "단어(공백으로 분리) → 바이트 시퀀스"로 변환
        word_freqs: Counter = Counter()
        for text in corpus:
            for word in self._pre_tokenize(text):
                word_freqs[word] += 1

        # 각 단어를 바이트 토큰 리스트로 표현
        # word_to_tokens: {word_string: [token_str, ...]}
        word_to_tokens: dict[str, list[str]] = {}
        for word in word_freqs:
            bs = word.encode("utf-8")
            word_to_tokens[word] = [self._byte_to_token(b) for b in bs]

        # 병합 루프
        self.merges = []
        target_merges = vocab_size - len(self.id_to_token)
        if target_merges <= 0:
            return

        for _ in range(target_merges):
            # 인접 쌍 빈도 집계
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                toks = word_to_tokens[word]
                for i in range(len(toks) - 1):
                    pair_freqs[(toks[i], toks[i + 1])] += freq
            if not pair_freqs:
                break
            # 가장 빈도 높은 쌍 선택 (동점이면 사전순으로 안정화)
            best_pair, best_freq = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))
            if best_freq < 2:
                # 의미 있는 병합이 더 이상 없음
                break
            # 새 토큰 생성 및 등록
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.id_to_token.append(new_token)
            self.token_to_id[new_token] = len(self.id_to_token) - 1
            # 모든 단어에서 해당 쌍을 병합
            for word in word_to_tokens:
                word_to_tokens[word] = self._merge_in_list(
                    word_to_tokens[word], best_pair, new_token
                )

        # 병합 순위 사전
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}

    # ---------- 인코딩 ----------

    def _tokenize(self, text: str) -> list[str]:
        """학습된 BPE 규칙으로 텍스트를 토큰화."""
        tokens: list[str] = []
        for word in self._pre_tokenize(text):
            # 단어를 바이트 토큰으로 분해
            bs = word.encode("utf-8")
            word_tokens = [self._byte_to_token(b) for b in bs]
            # 병합 규칙 적용 (rank가 낮은 순서로)
            word_tokens = self._apply_merges(word_tokens)
            tokens.extend(word_tokens)
        return tokens

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        """병합 규칙을 반복 적용하여 가장 긴 토큰들을 만듦."""
        if len(tokens) < 2:
            return tokens
        changed = True
        while changed:
            changed = False
            # 인접 쌍 중 가장 rank가 낮은(우선순위 높은) 쌍을 찾음
            best_idx = -1
            best_rank = len(self.merges) + 1
            for i in range(len(tokens) - 1):
                rank = self._merge_rank.get((tokens[i], tokens[i + 1]), -1)
                if rank >= 0 and rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx >= 0:
                # 병합
                merged = tokens[best_idx] + tokens[best_idx + 1]
                tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]
                changed = True
        return tokens

    # ---------- 전처리/후처리 ----------

    def _pre_tokenize(self, text: str) -> list[str]:
        """텍스트를 "단어" 단위로 분할.

        GPT-2 스타일: 공백을 단어 앞에 붙이고, 단어는 '▁' + 문자열 형태.
        여기서는 단순화하여 공백을 U+2581으로 치환 후 split.
        """
        # 모든 공백을 ▁(U+2581)으로 치환
        text = text.replace(" ", "▁")
        # 줄바꿈도 ▁으로 (단순화)
        text = text.replace("\n", "▁").replace("\t", "▁")
        # 연속 ▁을 하나로
        while "▁▁" in text:
            text = text.replace("▁▁", "▁")
        # ▁로 시작하면 분리
        if not text.startswith("▁"):
            text = "▁" + text
        # ▁을 기준으로 분할하지만 ▁을 유지
        parts = text.split("▁")
        words = []
        for i, p in enumerate(parts):
            if i == 0:
                continue  # 첫 번째는 빈 문자열
            if p:
                words.append("▁" + p)
        return words

    def _postprocess(self, text: str) -> str:
        """디코딩 후 ▁을 공백으로 복원."""
        # BPE는 바이트 단위로 토큰을 분해하므로, 디코딩 후 ▁이 여러 바이트 토큰으로
        # 쪼개져 있을 수 있음. 먼저 raw bytes에서 ▁을 공백으로 치환.
        # ▁(U+2581)의 UTF-8 인코딩은 0xE2 0x96 0x81 (3바이트)
        text = text.replace("▁", " ")
        # 양끝 공백 제거
        return text.strip()

    def _tokens_to_bytes(self, tokens: list[str]) -> bytes:
        """BPE 토큰을 바이트로 변환.

        각 토큰은 chr(byte)들의 시퀀스이므로, 각 문자의 코드포인트를
        바이트로 취급합니다 (Latin-1과 동일).
        """
        out = bytearray()
        for tok in tokens:
            for c in tok:
                out.append(ord(c) & 0xFF)
        return bytes(out)

    @staticmethod
    def _byte_to_token(b: int) -> str:
        """바이트를 토큰 문자열로 변환. chr(b)를 사용 (Latin-1 범위)."""
        return chr(b)

    @staticmethod
    def _merge_in_list(tokens: list[str], pair: tuple[str, str], new_token: str) -> list[str]:
        """tokens 리스트에서 pair를 new_token으로 병합."""
        result: list[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    # ---------- 저장/로드 ----------

    def _save_extra(self) -> dict:
        return {
            "merges": [[a, b] for a, b in self.merges],
        }

    def _load_extra(self, extra: dict) -> None:
        self.merges = [(a, b) for a, b in extra.get("merges", [])]
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}
