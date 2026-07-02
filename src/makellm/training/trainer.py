"""트레이너 — 학습 루프를 캡슐화."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..model.gpt import GPT
from ..utils.config import TrainerConfig, ModelConfig
from ..utils.logging import Logger
from ..utils.seed import set_seed
from .dataset import LMDataset, collate_batch
from .loss import cross_entropy_loss, compute_perplexity
from .optimizers import build_optimizer, build_scheduler


class Trainer:
    """GPT 모델 학습 루프.

    기능:
      - 에포크/스텝 반복
      - warmup + cosine LR 스케줄
      - 그래디언트 클리핑
      - 체크포인트 저장/로드
      - 주기적 로깅 (loss, lr, ppl)
    """

    def __init__(
        self,
        model: GPT,
        train_dataset: LMDataset,
        config: TrainerConfig,
        model_config: ModelConfig,
        eval_dataset: LMDataset | None = None,
        logger: Logger | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.model_config = model_config
        self.logger = logger or Logger("trainer", verbose=True)

        set_seed(config.seed)
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # 데이터로더
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_batch(b, pad_id=model_config.pad_token_id),
        )
        self.eval_loader = None
        if eval_dataset is not None:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_batch(b, pad_id=model_config.pad_token_id),
            )

        # 옵티마이저 & 스케줄러
        total_steps = self._estimate_total_steps()
        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(self.optimizer, config, total_steps)
        self.total_steps = total_steps
        self.global_step = 0

        # 출력 디렉토리
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _estimate_total_steps(self) -> int:
        """총 스텝 수 추정."""
        if self.config.max_steps is not None:
            return self.config.max_steps
        steps_per_epoch = max(1, len(self.train_dataset) // self.config.batch_size)
        return steps_per_epoch * self.config.max_epochs

    def train(self) -> dict:
        """학습 실행. 최종 메트릭 딕셔너리 반환."""
        self.logger.info(f"Starting training: {self.total_steps} steps, device={self.device}")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"Train dataset: {len(self.train_dataset)} samples, "
                         f"{self.train_dataset.total_tokens} tokens")

        self.model.train()
        t0 = time.time()
        history: list[dict] = []

        epoch = 0
        while self.global_step < self.total_steps:
            epoch += 1
            for inputs, targets in self.train_loader:
                if self.global_step >= self.total_steps:
                    break
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 순전파
                logits = self.model(inputs)
                loss = cross_entropy_loss(
                    logits, targets, ignore_index=self.model_config.pad_token_id
                )

                # 역전파
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # 그래디언트 클리핑
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.optimizer.step()
                self.scheduler.step()

                self.global_step += 1

                # 로깅
                if self.global_step % self.config.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.logger.step(
                        self.global_step, self.total_steps,
                        loss=loss.item(), lr=lr,
                        ppl=compute_perplexity(loss.item()),
                    )
                    history.append({
                        "step": self.global_step,
                        "loss": loss.item(),
                        "lr": lr,
                        "ppl": compute_perplexity(loss.item()),
                    })

                # 평가
                if self.eval_loader is not None and self.global_step % self.config.eval_every == 0:
                    eval_loss = self.evaluate()
                    self.logger.info(f"Eval at step {self.global_step}: loss={eval_loss:.4f}, "
                                     f"ppl={compute_perplexity(eval_loss):.2f}")

                # 체크포인트
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")

        elapsed = time.time() - t0
        self.logger.info(f"Training done in {elapsed:.1f}s. Final step={self.global_step}")
        self.save_checkpoint("final.pt")
        return {
            "history": history,
            "final_step": self.global_step,
            "elapsed_sec": elapsed,
        }

    def evaluate(self) -> float:
        """평가 데이터셋으로 손실 계산."""
        if self.eval_loader is None:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for inputs, targets in self.eval_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(inputs)
                loss = cross_entropy_loss(
                    logits, targets, ignore_index=self.model_config.pad_token_id
                )
                total_loss += loss.item()
                n_batches += 1
        self.model.train()
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, filename: str) -> None:
        """체크포인트 저장."""
        path = self.out_dir / filename
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "model_config": self.model_config.to_dict(),
            "trainer_config": self.config.to_dict(),
        }, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """체크포인트 로드."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step = ckpt["global_step"]
        self.logger.info(f"Loaded checkpoint: {path} (step={self.global_step})")
