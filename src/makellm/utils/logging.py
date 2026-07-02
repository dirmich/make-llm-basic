"""간단한 로깅 유틸 — print를 예쁘게 포장."""

from __future__ import annotations

import sys
import time
from typing import Any


class Logger:
    """최소한의 로거. tqdm 없이도 진행 상황을 표시."""

    def __init__(self, name: str = "makellm", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self._start = time.time()

    def _fmt(self, level: str, msg: str) -> str:
        elapsed = time.time() - self._start
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        return f"[{h:02d}:{m:02d}:{s:02d}] {level} | {msg}"

    def info(self, msg: Any) -> None:
        if self.verbose:
            print(self._fmt("INFO", str(msg)), flush=True)

    def warn(self, msg: Any) -> None:
        print(self._fmt("WARN", str(msg)), file=sys.stderr, flush=True)

    def error(self, msg: Any) -> None:
        print(self._fmt("ERR ", str(msg)), file=sys.stderr, flush=True)

    def step(self, step: int, total: int | None, **metrics: float) -> None:
        """학습 스텝 로그. metrics는 loss/lr 등."""
        if not self.verbose:
            return
        parts = [f"step={step}"]
        if total is not None:
            parts[0] = f"step={step}/{total}"
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(self._fmt("STEP", "  ".join(parts)), flush=True)
