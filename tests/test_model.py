"""모델 모듈 테스트 — 임베딩, 어텐션, 트랜스포머 블록, GPT."""

import pytest
import torch

from makellm.utils.config import ModelConfig
from makellm.model.embedding import TokenEmbedding, PositionalEmbedding, EmbeddingStack
from makellm.model.attention import (
    make_causal_mask,
    ScaledDotProductAttention,
    MultiHeadAttention,
)
from makellm.model.transformer_block import TransformerBlock, FeedForward
from makellm.model.gpt import GPT


@pytest.fixture
def small_config():
    return ModelConfig(
        vocab_size=100,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
    )


class TestEmbedding:
    def test_token_embedding_shape(self, small_config):
        emb = TokenEmbedding(small_config.vocab_size, small_config.d_model)
        ids = torch.randint(0, small_config.vocab_size, (2, 8))
        out = emb(ids)
        assert out.shape == (2, 8, small_config.d_model)

    def test_positional_sinusoidal(self, small_config):
        pe = PositionalEmbedding(small_config.d_model, max_len=64, mode="sinusoidal")
        out = pe(seq_len=16)
        assert out.shape == (1, 16, small_config.d_model)

    def test_positional_learned(self, small_config):
        pe = PositionalEmbedding(small_config.d_model, max_len=64, mode="learned")
        out = pe(seq_len=16)
        assert out.shape == (1, 16, small_config.d_model)

    def test_positional_exceeds_max_len_raises(self, small_config):
        pe = PositionalEmbedding(small_config.d_model, max_len=32, mode="learned")
        with pytest.raises(ValueError):
            pe(seq_len=64)

    def test_embedding_stack(self, small_config):
        stack = EmbeddingStack(
            small_config.vocab_size, small_config.d_model,
            max_len=64, dropout=0.0
        )
        ids = torch.randint(0, small_config.vocab_size, (2, 8))
        out = stack(ids)
        assert out.shape == (2, 8, small_config.d_model)


class TestAttention:
    def test_causal_mask_shape(self):
        m = make_causal_mask(8)
        assert m.shape == (8, 8)

    def test_causal_mask_blocks_future(self):
        """마스크의 상삼각(주대각선 위)은 -inf, 하삼각은 0이어야 함."""
        m = make_causal_mask(4)
        # 주대각선 위는 -inf
        assert m[0, 1] == float("-inf")
        assert m[0, 2] == float("-inf")
        assert m[1, 2] == float("-inf")
        # 주대각선과 아래는 0
        assert m[0, 0] == 0.0
        assert m[1, 0] == 0.0
        assert m[1, 1] == 0.0

    def test_scaled_dot_product_attention(self, small_config):
        attn = ScaledDotProductAttention()
        q = torch.randn(2, 4, 8, 8)  # [batch, heads, seq, d_head]
        k = torch.randn(2, 4, 8, 8)
        v = torch.randn(2, 4, 8, 8)
        out = attn(q, k, v)
        assert out.shape == (2, 4, 8, 8)

    def test_multihead_attention_shape(self, small_config):
        attn = MultiHeadAttention(small_config)
        x = torch.randn(2, 8, small_config.d_model)
        out = attn(x)
        assert out.shape == (2, 8, small_config.d_model)

    def test_multihead_attention_with_mask(self, small_config):
        attn = MultiHeadAttention(small_config)
        x = torch.randn(2, 8, small_config.d_model)
        mask = make_causal_mask(8)
        out = attn(x, mask=mask)
        assert out.shape == (2, 8, small_config.d_model)

    def test_attention_is_causal(self, small_config):
        """인과 마스크를 쓰면 위치 i의 출력이 위치 i+1 이후에 영향을 받지 않아야 함."""
        attn = MultiHeadAttention(small_config)
        attn.eval()
        x = torch.randn(1, 6, small_config.d_model)
        mask = make_causal_mask(6)
        out1 = attn(x, mask=mask)
        # x의 마지막 위치를 바꿔도 앞 위치의 출력은 변하지 않아야 함
        x2 = x.clone()
        x2[:, -1] = torch.randn(1, small_config.d_model)
        out2 = attn(x2, mask=mask)
        # 앞 5개 위치는 동일해야 함
        assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-5)


class TestTransformerBlock:
    def test_block_shape(self, small_config):
        block = TransformerBlock(small_config)
        x = torch.randn(2, 8, small_config.d_model)
        out = block(x)
        assert out.shape == (2, 8, small_config.d_model)

    def test_block_residual_connection(self, small_config):
        """잔차 연결이 있으면 입력과 출력의 차이가 유한해야 함."""
        block = TransformerBlock(small_config)
        block.eval()
        x = torch.randn(2, 8, small_config.d_model)
        out = block(x)
        # 출력이 입력과 비슷한 스케일이어야 함 (잔차)
        assert out.shape == x.shape
        diff = (out - x).abs().mean()
        assert diff.item() < 5.0  # 잔차가 너무 크면 정규화 문제

    def test_feedforward_shape(self, small_config):
        ffn = FeedForward(small_config.d_model, small_config.d_ff)
        x = torch.randn(2, 8, small_config.d_model)
        out = ffn(x)
        assert out.shape == (2, 8, small_config.d_model)


class TestGPT:
    def test_gpt_forward_shape(self, small_config):
        model = GPT(small_config)
        ids = torch.randint(0, small_config.vocab_size, (2, 8))
        logits = model(ids)
        assert logits.shape == (2, 8, small_config.vocab_size)

    def test_gpt_param_count(self, small_config):
        model = GPT(small_config)
        n = model.num_parameters()
        assert n > 0
        # 2-layer, 32-dim, 100-vocab 모델은 약 30K~50K 파라미터
        assert 10_000 < n < 200_000

    def test_gpt_returns_hidden(self, small_config):
        model = GPT(small_config)
        ids = torch.randint(0, small_config.vocab_size, (2, 8))
        logits, hidden = model(ids, return_hidden=True)
        assert logits.shape == (2, 8, small_config.vocab_size)
        assert hidden.shape == (2, 8, small_config.d_model)

    def test_gpt_weight_tying(self, small_config):
        """LM head 가중치가 토큰 임베딩과 공유되는지 확인."""
        model = GPT(small_config)
        assert model.lm_head.weight is model.embedding.token_emb.embedding.weight

    def test_gpt_context_length_limit(self, small_config):
        """컨텍스트 길이를 초과하면 에러."""
        model = GPT(small_config)
        ids = torch.randint(0, small_config.vocab_size, (1, small_config.context_length + 1))
        # 임베딩 단계에서 에러 발생해야 함
        with pytest.raises(ValueError):
            _ = model(ids)

    def test_gpt_no_grad_in_eval(self, small_config):
        """eval 모드에서 dropout이 꺼지는지 확인 (출력 재현성)."""
        model = GPT(small_config)
        model.eval()
        ids = torch.randint(0, small_config.vocab_size, (1, 8))
        out1 = model(ids)
        out2 = model(ids)
        assert torch.allclose(out1, out2, atol=1e-6)
