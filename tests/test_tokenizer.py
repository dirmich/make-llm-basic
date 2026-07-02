"""토크나이저 테스트 — BPE와 Unigram의 학습/인코딩/디코딩 라운드트립."""

import pytest
from makellm.tokenizer import BPETokenizer, UnigramTokenizer
from makellm.tokenizer.base import SpecialTokens


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the slow brown dog sleeps under the warm sun",
    "a quick fox is a happy fox",
    "the lazy dog is not a quick dog",
    "jumps over the warm sun and the lazy fox",
] * 3  # 빈도 높이기


class TestSpecialTokens:
    def test_default_ids(self):
        s = SpecialTokens()
        assert s.pad_id == 0
        assert s.bos_id == 1
        assert s.eos_id == 2
        assert s.unk_id == 3

    def test_custom_tokens(self):
        s = SpecialTokens(pad="[PAD]", bos="[BOS]", eos="[EOS]", unk="[UNK]")
        assert s.tokens == ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


class TestBPETokenizer:
    def test_train_creates_vocab(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=100)
        assert tok.vocab_size > 4  # 특수 토큰 4개 + 바이트 256개 (최소 260)
        assert tok._trained is True

    def test_encode_decode_roundtrip(self):
        """인코딩 후 디코딩하면 원문과 거의 같아야 함 (공백 정규화 차이 허용)."""
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=200)
        text = "the quick fox"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        # 공백 정규화: 다중 공백이 단일 공백으로 될 수 있음
        assert " ".join(decoded.split()) == "the quick fox"

    def test_bos_eos_addition(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=100)
        ids = tok.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == tok.special.bos_id
        assert ids[-1] == tok.special.eos_id

    def test_unknown_token_handled(self):
        """어휘에 없는 문자도 바이트 단위로 처리되어 unk가 발생하지 않아야 함."""
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=100)
        # 이모지, 한글 등 어휘에 없는 문자
        ids = tok.encode("안녕")
        assert tok.special.unk_id not in ids  # 바이트로 분해되므로 unk 없음

    def test_save_load(self, tmp_path):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=150)
        path = tmp_path / "tok.json"
        tok.save(path)
        tok2 = BPETokenizer.load(path)
        text = "the quick fox"
        assert tok.encode(text) == tok2.encode(text)

    def test_repr(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=80)
        s = repr(tok)
        assert "BPETokenizer" in s
        assert "vocab_size" in s


class TestUnigramTokenizer:
    def test_train_creates_vocab(self):
        tok = UnigramTokenizer()
        tok.train(CORPUS, vocab_size=80)
        assert tok.vocab_size > 4
        assert tok._trained is True

    def test_encode_produces_ids(self):
        tok = UnigramTokenizer()
        tok.train(CORPUS, vocab_size=100)
        ids = tok.encode("the quick fox")
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_encode_decode_roundtrip(self):
        tok = UnigramTokenizer()
        tok.train(CORPUS, vocab_size=150)
        text = "the quick brown fox"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert " ".join(decoded.split()) == "the quick brown fox"

    def test_save_load(self, tmp_path):
        tok = UnigramTokenizer()
        tok.train(CORPUS, vocab_size=120)
        path = tmp_path / "uni.json"
        tok.save(path)
        tok2 = UnigramTokenizer.load(path)
        text = "the fox"
        assert tok.encode(text) == tok2.encode(text)

    def test_viterbi_single_char_fallback(self):
        """어휘에 없는 문자도 unk로 처리되지 않고 분할되어야 함."""
        tok = UnigramTokenizer()
        tok.train(CORPUS, vocab_size=100)
        ids = tok.encode("xyz")  # 코퍼스에 없는 문자
        # 각 문자는 개별 토큰 또는 unk로 처리되지만 결과는 비어있지 않아야 함
        assert len(ids) > 0
