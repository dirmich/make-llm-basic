# Make LLM-basic

**기초부터 직접 만드는 대형 언어 모델**

저자: dirmich | 출판사: Highmaru Press

## 소개

이 패키지는 **Make LLM-basic** 책의 전체 소스 코드와 LaTeX 원고를 포함합니다. 딥러닝 초보자가 토크나이저부터 텍스트 생성까지 LLM의 모든 구성 요소를 직접 구현해보는 것을 목표로 합니다.

## 디렉토리 구조

```
make-llm-basic/
├── src/makellm/              # LLM 구현 패키지
│   ├── tokenizer/            # BPE, Unigram 토크나이저
│   ├── model/                # 임베딩, 어텐션, 트랜스포머, GPT
│   ├── training/             # 데이터셋, 손실, 옵티마이저, 트레이너
│   ├── inference/            # 샘플러, 생성
│   └── utils/                # 설정, 시드, 로깅
├── tests/                    # pytest 단위 테스트 (48개)
├── book/                     # LaTeX 원고
│   ├── main.tex              # 메인 파일
│   ├── shared/preamble.tex   # 공유 프리앰블
│   ├── chapters/             # 각 장 .tex 파일
│   └── make_llm_basic.pdf    # 컴파일된 PDF
├── pyproject.toml
├── requirements.txt
└── README.md (이 파일)
```

## 설치

```bash
cd make-llm-basic
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

## 테스트 실행

```bash
pytest tests/ -v
```

모든 48개 테스트가 통과해야 합니다.

## 빠른 시작

```python
from makellm.tokenizer import BPETokenizer
from makellm.model import GPT
from makellm.training import LMDataset, Trainer
from makellm.inference import generate, TopPSampler
from makellm.utils import ModelConfig, TrainerConfig, set_seed

set_seed(42)

# 토크나이저 학습
corpus = ["the quick brown fox jumps", "the lazy dog sleeps"] * 100
tokenizer = BPETokenizer()
tokenizer.train(corpus, vocab_size=500)

# 모델 생성
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    context_length=32, d_model=64, n_heads=4, n_layers=2, d_ff=128
)
model = GPT(config)
print(f"Parameters: {model.num_parameters():,}")

# 학습
dataset = LMDataset(corpus, tokenizer, context_length=32)
trainer = Trainer(model, dataset, TrainerConfig(max_steps=100), config)
trainer.train()

# 생성
text = generate(model, tokenizer, prompt="the", max_new_tokens=20,
                sampler=TopPSampler(p=0.9, temperature=0.8))
print(f"Generated: {text}")
```

## 책 PDF 빌드

PDF는 Tectonic으로 컴파일할 수 있습니다.

```bash
cd book
tectonic -o . --outname make_llm_basic main.tex
```

## 라이선스

- 코드: MIT
- 책 본문: CC BY-NC-SA 4.0

## 관련 자료

- 2권: Make LLM-advanced (분산 학습, 양자화, RLHF 등)
- PRD/Plan/Task 문서: 별도 PDF
