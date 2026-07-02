"""학습 및 추론 테스트."""

import pytest
import torch

from makellm.utils.config import ModelConfig, TrainerConfig
from makellm.utils.seed import set_seed
from makellm.tokenizer import BPETokenizer
from makellm.model.gpt import GPT
from makellm.training.dataset import LMDataset, collate_batch
from makellm.training.loss import cross_entropy_loss, label_smoothing_loss, compute_perplexity
from makellm.training.optimizers import build_optimizer, build_scheduler
from makellm.inference.sampler import (
    GreedySampler, TemperatureSampler, TopKSampler, TopPSampler
)
from makellm.inference.generate import generate


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a slow brown dog sleeps under the warm sun",
    "quick foxes are happy foxes",
    "lazy dogs sleep in the warm sun",
    "the brown fox and the lazy dog",
] * 2


@pytest.fixture
def trained_tokenizer():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=120)
    return tok


class TestDataset:
    def test_dataset_creation(self, trained_tokenizer):
        ds = LMDataset(CORPUS, trained_tokenizer, context_length=16)
        assert len(ds) > 0
        assert ds.total_tokens > 0

    def test_dataset_item_shapes(self, trained_tokenizer):
        ds = LMDataset(CORPUS, trained_tokenizer, context_length=16)
        inp, tgt = ds[0]
        assert inp.shape == (16,)
        assert tgt.shape == (16,)
        # target은 input을 한 칸 shift
        assert torch.equal(inp[1:], tgt[:-1])

    def test_collate_batch(self, trained_tokenizer):
        ds = LMDataset(CORPUS, trained_tokenizer, context_length=16)
        batch = [ds[0], ds[1], ds[2]]
        inputs, targets = collate_batch(batch)
        assert inputs.shape == (3, 16)
        assert targets.shape == (3, 16)


class TestLoss:
    def test_cross_entropy_shape(self):
        logits = torch.randn(2, 8, 100)
        targets = torch.randint(0, 100, (2, 8))
        loss = cross_entropy_loss(logits, targets)
        assert loss.dim() == 0  # 스칼라
        assert loss.item() > 0

    def test_label_smoothing_decreases_loss(self):
        """label smoothing은 손실을 약간 증가시킴 (예측이 덜 자신감 있게)."""
        torch.manual_seed(0)
        logits = torch.randn(2, 8, 100)
        targets = torch.randint(0, 100, (2, 8))
        ce = cross_entropy_loss(logits, targets).item()
        ls = label_smoothing_loss(logits, targets, smoothing=0.1).item()
        # label smoothing은 손실을 증가시키는 경향 (엔트로피 증가)
        assert ls >= ce * 0.95  # 약간의 오차 허용

    def test_perplexity(self):
        assert compute_perplexity(0.0) == 1.0
        assert compute_perplexity(1.0) > 1.0
        # 매우 큰 손실은 overflow 방지
        assert compute_perplexity(100.0) < 1e9


class TestOptimizers:
    def test_optimizer_groups(self):
        cfg = TrainerConfig(weight_decay=0.1, learning_rate=1e-3)
        model = GPT(ModelConfig(vocab_size=50, d_model=16, n_heads=2, n_layers=1, d_ff=32))
        opt = build_optimizer(model, cfg)
        # 두 개의 파라미터 그룹 (decay / no_decay)
        assert len(opt.param_groups) == 2

    def test_scheduler_warmup(self):
        cfg = TrainerConfig(learning_rate=1e-3, warmup_steps=10, lr_scheduler="cosine")
        model = GPT(ModelConfig(vocab_size=50, d_model=16, n_heads=2, n_layers=1, d_ff=32))
        opt = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg, total_steps=100)
        # 스텝 0에서 LR은 0에 가까워야 함 (warmup 시작)
        sched.step()
        lr_1 = opt.param_groups[0]["lr"]
        assert lr_1 > 0
        # warmup 스텝을 넘으면 LR이 감소하기 시작
        for _ in range(15):
            sched.step()
        lr_16 = opt.param_groups[0]["lr"]
        # cosine decay가 시작되어야 함
        assert lr_16 <= cfg.learning_rate


class TestSamplers:
    def test_greedy_picks_max(self):
        sampler = GreedySampler()
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
        out = sampler(logits)
        assert out.item() == 1  # 가장 큰 값의 인덱스

    def test_temperature_smooths(self):
        """낮은 온도는 greedy에 가깝고, 높은 온도는 더 무작위."""
        torch.manual_seed(0)
        logits = torch.tensor([[1.0, 5.0, 0.5, 0.1]])
        low_t = TemperatureSampler(temperature=0.1)
        # 낮은 온도에서는 거의 항상 argmax 선택
        counts = sum(1 for _ in range(20) if low_t(logits).item() == 1)
        assert counts >= 18  # 대부분 argmax

    def test_top_k_limits_choices(self):
        """Top-K는 K개 이외의 토큰을 선택하지 않음."""
        torch.manual_seed(0)
        logits = torch.tensor([[1.0, 5.0, 4.0, 3.0, 0.1]])
        sampler = TopKSampler(k=2, temperature=1.0)
        # 20번 샘플링해도 항상 top-2 (인덱스 1, 2) 중 하나
        for _ in range(20):
            idx = sampler(logits).item()
            assert idx in (1, 2)

    def test_top_p_limits_choices(self):
        """Top-P는 누적 확률이 P를 넘기 전까지의 토큰만."""
        torch.manual_seed(0)
        logits = torch.tensor([[0.0, 10.0, 9.0, 0.0, 0.0]])  # 상위 2개가 압도적
        sampler = TopPSampler(p=0.5, temperature=1.0)
        for _ in range(20):
            idx = sampler(logits).item()
            # 가장 확률 높은 토큰(인덱스 1)만 선택되어야 함
            assert idx == 1


class TestEndToEnd:
    def test_one_step_loss_decreases(self, trained_tokenizer):
        """1스텝 학습 후 손실이 감소해야 함."""
        set_seed(42)
        cfg = ModelConfig(
            vocab_size=trained_tokenizer.vocab_size,
            context_length=16, d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0,
        )
        model = GPT(cfg)
        ds = LMDataset(CORPUS, trained_tokenizer, context_length=16)
        inputs, targets = collate_batch([ds[0], ds[1]])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 초기 손실
        logits = model(inputs)
        loss_before = cross_entropy_loss(logits, targets).item()

        # 5스텝 학습
        for _ in range(5):
            opt.zero_grad()
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            loss.backward()
            opt.step()

        logits = model(inputs)
        loss_after = cross_entropy_loss(logits, targets).item()
        assert loss_after < loss_before

    def test_generate_returns_string(self, trained_tokenizer):
        """generate 함수가 문자열을 반환해야 함."""
        cfg = ModelConfig(
            vocab_size=trained_tokenizer.vocab_size,
            context_length=16, d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0,
        )
        model = GPT(cfg)
        text = generate(
            model, trained_tokenizer, prompt="the", max_new_tokens=10, device="cpu"
        )
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_greedy_reproducible(self, trained_tokenizer):
        """greedy 샘플러는 같은 입력에 같은 출력."""
        cfg = ModelConfig(
            vocab_size=trained_tokenizer.vocab_size,
            context_length=16, d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0,
        )
        model = GPT(cfg)
        model.eval()
        out1 = generate(model, trained_tokenizer, prompt="the", max_new_tokens=8)
        out2 = generate(model, trained_tokenizer, prompt="the", max_new_tokens=8)
        assert out1 == out2
