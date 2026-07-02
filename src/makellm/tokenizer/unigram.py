"""Unigram 토크나이저 (SentencePiece 방식).

EM(Expectation-Maximization)으로 어휘를 학습하고
Viterbi 알고리즘으로 최적 분할을 찾는 방식.

참고: Kudo (2018) "SentencePiece: A simple and language independent subword
tokenizer and detokenizer for Neural Text Processing"
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from .base import Tokenizer


class UnigramTokenizer(Tokenizer):
    """Unigram 언어 모델 기반 토크나이저.

    특징:
      - 각 토큰에 점수(log 확률) 할당
      - Viterbi로 텍스트의 최적 분할 탐색
      - EM으로 어휘 및 점수를 반복 개선
    """

    def __init__(self, special_tokens=None):
        super().__init__(special_tokens=special_tokens)
        self.token_scores: dict[str, float] = {}

    # ---------- 학습 ----------

    def _train(self, corpus: Sequence[str], vocab_size: int) -> None:
        """EM 기반 Unigram 어휘 학습.

        알고리즘:
          1. 코퍼스에서 모든 부분문자열을 추출하여 초기 후보 어휘 생성
          2. 각 토큰의 빈도를 확률로 변환
          3. EM 반복:
             - E-step: Viterbi로 각 단어의 최적 분할 찾기
             - M-step: 분할 결과로 토큰 확률 갱신
          4. 낮은 확률 토큰을 제거하며 어휘 축소
        """
        # 전처리: 공백을 ▁으로, 단어 단위 분할
        words: list[str] = []
        word_freqs: Counter = Counter()
        for text in corpus:
            for w in self._pre_tokenize(text):
                word_freqs[w] += 1
                if w not in words:
                    words.append(w)

        # 초기 후보: 모든 부분문자열 (길이 1~최대 16)
        substr_freqs: Counter = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word)):
                for j in range(i + 1, min(len(word), i + 16) + 1):
                    substr = word[i:j]
                    substr_freqs[substr] += freq

        # 초기 어휘: 빈도 순 상위 N개
        # 특수 토큰 4개 + 빈도 높은 부분문자열
        target = max(vocab_size, len(self.special.tokens) + 100)
        candidates = sorted(substr_freqs.items(), key=lambda x: -x[1])[:target]
        # 각 토큰의 log 확률 (총합으로 정규화)
        total = sum(f for _, f in candidates) or 1
        self.token_scores = {tok: math.log(freq / total) for tok, freq in candidates}
        # 어휘 리스트에 추가
        self.id_to_token = list(self.special.tokens)
        for tok in self.token_scores:
            if tok not in self.token_to_id:
                self.id_to_token.append(tok)
                self.token_to_id[tok] = len(self.id_to_token) - 1

        # EM 반복 (2-3회면 충분)
        for _ in range(2):
            self._em_step(words, word_freqs)

        # 어휘 축소: 목표 크기까지 낮은 점수 토큰 제거
        while len(self.token_scores) > (vocab_size - len(self.special.tokens)):
            # 가장 점수가 낮은 토큰 하나 제거
            worst = min(self.token_scores.items(), key=lambda x: x[1])
            del self.token_scores[worst[0]]

        # id_to_token 재구성
        self.id_to_token = list(self.special.tokens)
        for tok in self.token_scores:
            self.id_to_token.append(tok)
            self.token_to_id[tok] = len(self.id_to_token) - 1

    def _em_step(self, words: list[str], word_freqs: Counter) -> None:
        """EM 1스텝. 각 단어의 최적 분할을 찾고 토큰 점수를 갱신."""
        # E-step: 각 단어의 최적 분할
        token_total_freq: Counter = Counter()
        for word, freq in word_freqs.items():
            best_path, _ = self._viterbi(word)
            for tok in best_path:
                token_total_freq[tok] += freq
        # M-step: 점수 갱신
        total = sum(token_total_freq.values()) or 1
        for tok in list(self.token_scores.keys()):
            f = token_total_freq.get(tok, 0)
            # 0회 등장 토큰은 매우 낮은 점수 (제거 대상)
            self.token_scores[tok] = math.log(max(f, 1) / total) - 10.0 if f == 0 else math.log(f / total)

    def _viterbi(self, word: str) -> tuple[list[str], float]:
        """단어에 대한 최적 분할을 Viterbi로 탐색.

        Returns:
            (최적 토큰 리스트, 총 log 점수)
        """
        n = len(word)
        # dp[i] = (최대 점수, 이전 인덱스, 이전 토큰)
        dp: list[tuple[float, int, str]] = [(-float("inf"), -1, "") for _ in range(n + 1)]
        dp[0] = (0.0, -1, "")
        # 단일 문자 보정: 어휘에 없는 문자는 unk로 처리되지만 학습 중에는 매우 낮은 점수 부여
        for i in range(1, n + 1):
            for j in range(max(0, i - 16), i):
                substr = word[j:i]
                if substr in self.token_scores:
                    score = self.token_scores[substr]
                else:
                    # 어휘에 없는 부분문자열: 큰 패널티
                    score = -20.0
                candidate = dp[j][0] + score
                if candidate > dp[i][0]:
                    dp[i] = (candidate, j, substr)
        # 역추적
        path: list[str] = []
        i = n
        while i > 0:
            _, j, tok = dp[i]
            path.append(tok)
            i = j
        path.reverse()
        return path, dp[n][0]

    # ---------- 인코딩 ----------

    def _tokenize(self, text: str) -> list[str]:
        """Viterbi로 최적 분할 탐색."""
        tokens: list[str] = []
        for word in self._pre_tokenize(text):
            best_path, _ = self._viterbi(word)
            tokens.extend(best_path)
        return tokens

    # ---------- 전처리/후처리 ----------

    def _pre_tokenize(self, text: str) -> list[str]:
        """공백을 ▁으로 치환하고 단어 단위로 분할."""
        text = text.replace(" ", "▁")
        text = text.replace("\n", "▁").replace("\t", "▁")
        while "▁▁" in text:
            text = text.replace("▁▁", "▁")
        if not text.startswith("▁"):
            text = "▁" + text
        parts = text.split("▁")
        words = []
        for i, p in enumerate(parts):
            if i == 0:
                continue
            if p:
                words.append("▁" + p)
        return words

    def _postprocess(self, text: str) -> str:
        text = text.replace("▁", " ")
        return text.strip()

    # ---------- 저장/로드 ----------

    def _save_extra(self) -> dict:
        return {
            "scores": self.token_scores,
        }

    def _load_extra(self, extra: dict) -> None:
        # JSON 키는 항상 문자열이므로 그대로 로드
        self.token_scores = extra.get("scores", {})
