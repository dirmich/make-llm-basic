"""End-to-end 학습 데모: 작은 말뭉치로 GPT 학습 후 텍스트 생성.

이 스크립트는 1권의 모든 모듈을 결합하여:
  1. 작은 텍스트 말뭉치를 로드
  2. BPE 토크나이저 학습
  3. GPT 모델 구성
  4. 학습 루프 실행
  5. 다양한 샘플링 전략으로 텍스트 생성

실행:
    python scripts/tiny_train.py
    python scripts/tiny_train.py --epochs 5 --vocab 500
    python scripts/tiny_train.py --text-path my_corpus.txt

CPU에서 1~3분 내 실행 완료 (테스트된 구성 기준).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 패키지 경로 설정 (스크립트로 직접 실행 시)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from makellm.tokenizer import BPETokenizer
from makellm.model import GPT
from makellm.training import LMDataset, Trainer, cross_entropy_loss, compute_perplexity
from makellm.inference import (
    generate,
    GreedySampler,
    TemperatureSampler,
    TopKSampler,
    TopPSampler,
)
from makellm.utils import ModelConfig, TrainerConfig, set_seed, Logger


# ============ 기본 말뭉치 (텍스트 파일이 없을 때 사용) ============

DEFAULT_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a quick fox is a happy fox",
    "the lazy dog sleeps under the warm sun",
    "the brown fox and the lazy dog are friends",
    "foxes are quick and dogs are loyal",
    "the warm sun shines on the brown fox",
    "the quick fox jumps over the sleeping dog",
    "a happy fox plays in the warm sun",
    "the lazy dog watches the quick fox",
    "brown foxes and lazy dogs live together",
    "the sun is warm and the fox is quick",
    "the dog is lazy but the fox is fast",
    "quick brown foxes jump high",
    "lazy dogs sleep all day in the sun",
    "the fox and the dog are good friends",
] * 20  # 15 sentences * 20 = 300 sentences, ~5000 tokens


def load_corpus(text_path: str | None) -> list[str]:
    """말뭉치 로드. 파일이 없으면 기본 말뭉치 사용."""
    if text_path:
        path = Path(text_path)
        if not path.exists():
            print(f"Warning: {text_path} not found, using default corpus")
            return DEFAULT_CORPUS
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return DEFAULT_CORPUS


def train_tokenizer(corpus: list[str], vocab_size: int) -> BPETokenizer:
    """BPE 토크나이저 학습."""
    print("\n[1/5] 토크나이저 학습 중...")
    t0 = time.time()
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab_size)
    print(f"    완료: {tok.vocab_size}개 토큰, {time.time()-t0:.1f}초")
    # 샘플 인코딩 출력
    sample = corpus[0]
    ids = tok.encode(sample)
    print(f"    샘플: '{sample}'")
    print(f"    → {len(ids)}개 토큰: {ids[:20]}{'...' if len(ids) > 20 else ''}")
    return tok


def build_model(tokenizer: BPETokenizer, d_model: int, n_layers: int) -> tuple[GPT, ModelConfig]:
    """GPT 모델 구성."""
    print("\n[2/5] 모델 구성 중...")
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=64,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_ff=d_model * 4,
        dropout=0.1,
    )
    model = GPT(config)
    print(f"    완료: {model}")
    print(f"    파라미터 수: {model.num_parameters():,}")
    return model, config


def train_model(
    model: GPT,
    config: ModelConfig,
    tokenizer: BPETokenizer,
    corpus: list[str],
    epochs: int,
    batch_size: int,
) -> dict:
    """모델 학습."""
    print("\n[3/5] 모델 학습 중...")
    dataset = LMDataset(corpus, tokenizer, context_length=config.context_length)
    print(f"    데이터셋: {len(dataset)}개 샘플, {dataset.total_tokens}개 토큰")

    trainer_config = TrainerConfig(
        batch_size=batch_size,
        learning_rate=3e-4,
        max_epochs=epochs,
        warmup_steps=10,
        lr_scheduler="cosine",
        grad_clip=1.0,
        log_every=20,
        eval_every=100,
        save_every=10000,  # 자동 저장 안 함
        device="cpu",
        out_dir="./runs",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        config=trainer_config,
        model_config=config,
    )
    result = trainer.train()
    print(f"    학습 완료: {result['final_step']}스텝, {result['elapsed_sec']:.1f}초")
    return result


def generate_samples(model: GPT, tokenizer: BPETokenizer, prompts: list[str]) -> None:
    """다양한 샘플링 전략으로 텍스트 생성."""
    print("\n[4/5] 텍스트 생성 중...")
    samplers = [
        ("Greedy", GreedySampler()),
        ("Temperature=0.5", TemperatureSampler(temperature=0.5)),
        ("Temperature=1.0", TemperatureSampler(temperature=1.0)),
        ("Top-K=10", TopKSampler(k=10, temperature=0.8)),
        ("Top-P=0.9", TopPSampler(p=0.9, temperature=0.8)),
    ]

    for prompt in prompts:
        print(f"\n  프롬프트: '{prompt}'")
        for name, sampler in samplers:
            text = generate(
                model, tokenizer, prompt=prompt,
                max_new_tokens=20, sampler=sampler,
                device="cpu",
            )
            # 생성된 부분만 표시
            generated = text[len(prompt):].strip() if text.startswith(prompt) else text
            print(f"    [{name:20s}] {generated}")


def evaluate_model(model: GPT, config: ModelConfig, tokenizer: BPETokenizer, corpus: list[str]) -> float:
    """모델 평가 (perplexity)."""
    print("\n[5/5] 모델 평가 중...")
    model.eval()
    dataset = LMDataset(corpus, tokenizer, context_length=config.context_length)
    total_loss = 0.0
    n = 0
    import torch
    with torch.no_grad():
        for i in range(min(20, len(dataset))):
            inputs, targets = dataset[i]
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets, ignore_index=config.pad_token_id)
            total_loss += loss.item()
            n += 1
    avg_loss = total_loss / max(n, 1)
    ppl = compute_perplexity(avg_loss)
    print(f"    평균 손실: {avg_loss:.4f}")
    print(f"    Perplexity: {ppl:.2f}")
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Make LLM-basic: tiny training demo")
    parser.add_argument("--text-path", type=str, default=None,
                        help="말뭉치 텍스트 파일 경로 (한 줄에 한 문장)")
    parser.add_argument("--vocab", type=int, default=500,
                        help="토크나이저 어휘 크기")
    parser.add_argument("--d-model", type=int, default=64,
                        help="모델 임베딩 차원")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="트랜스포머 블록 수")
    parser.add_argument("--epochs", type=int, default=3,
                        help="학습 에포크 수")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="배치 크기")
    args = parser.parse_args()

    print("=" * 60)
    print("  Make LLM-basic: End-to-End Training Demo")
    print("=" * 60)

    set_seed(42)

    # 1. 말뭉치 로드
    corpus = load_corpus(args.text_path)
    print(f"\n말뭉치: {len(corpus)}문장, {sum(len(s) for s in corpus):,}문자")

    # 2. 토크나이저 학습
    tokenizer = train_tokenizer(corpus, args.vocab)

    # 3. 모델 구성
    model, config = build_model(tokenizer, args.d_model, args.n_layers)

    # 4. 학습
    train_model(model, config, tokenizer, corpus, args.epochs, args.batch_size)

    # 5. 평가
    evaluate_model(model, config, tokenizer, corpus)

    # 6. 샘플 생성
    generate_samples(model, tokenizer, prompts=["the", "a quick", "the lazy"])

    print("\n" + "=" * 60)
    print("  데모 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("  - args.epochs를 늘려 더 오래 학습해보세요")
    print("  - args.d-model과 args.n-layers를 키워 더 큰 모델을 시도해보세요")
    print("  - --text-path로 자신만의 말뭉치를 학습해보세요")


if __name__ == "__main__":
    main()
