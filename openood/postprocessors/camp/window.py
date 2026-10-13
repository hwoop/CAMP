from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class WindowRows:
    features: torch.Tensor         # [n, d] float32
    msp: torch.Tensor              # [n]    float64
    pseudo_class: torch.Tensor     # [n]    int64
    semantic_logit: torch.Tensor   # [n]    float64

    @property
    def count(self) -> int:
        return int(self.features.shape[0])


class StreamWindow:
    """
    Sliding window over the most recent ``window_size`` samples of the stream.
    ``ready`` becomes true once ``min_samples`` samples have been collected.
    """

    COMPACT_CHUNKS = 64

    def __init__(
        self,
        window_size: int,
        min_samples: int,
    ) -> None:

        if int(window_size) <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}.")

        if int(min_samples) < 2:
            raise ValueError(f"min_samples must be >= 2, got {min_samples}.")

        self.window_size = int(window_size)
        self.warmup = max(2, min(int(min_samples), self.window_size))
        self.reset()


    def reset(self) -> None:

        self._chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._count = 0
        self.seen = 0


    @property
    def count(self) -> int:
        return self._count


    @property
    def ready(self) -> bool:
        return self._count >= self.warmup


    def push(
        self,
        features: torch.Tensor,
        msp: torch.Tensor,
        pseudo_class: torch.Tensor,
        semantic_logit: torch.Tensor,
    ) -> None:

        feat = features.detach().to(torch.float32)
        columns = (
            feat,
            msp.detach().to(feat.device, torch.float64),
            pseudo_class.detach().to(feat.device, torch.long),
            semantic_logit.detach().to(torch.float64).reshape(-1),
        )

        n = int(feat.shape[0])
        if any(int(column.shape[0]) != n for column in columns):
            raise ValueError("the columns of one batch must have the same number of rows.")

        if n == 0:
            return

        self._chunks.append(columns)
        self._count += n
        self.seen += n

        excess = self._count - self.window_size
        while excess > 0 and self._chunks:
            head = self._chunks[0]
            size = int(head[0].shape[0])
            if size <= excess:
                self._chunks.pop(0)
                self._count -= size
                excess -= size
            else:
                self._chunks[0] = tuple(column[excess:] for column in head)
                self._count -= excess
                excess = 0

        if len(self._chunks) > self.COMPACT_CHUNKS:
            self._chunks = [tuple(torch.cat(column) for column in zip(*self._chunks))]


    def rows(self) -> WindowRows | None:

        if not self._chunks:
            return None

        if len(self._chunks) == 1:
            return WindowRows(*self._chunks[0])

        rows = [torch.cat(column) for column in zip(*self._chunks)]
        return WindowRows(*rows)


    def state_bytes(self) -> int:

        bytes = sum(
            column.numel() * column.element_size()
            for chunk in self._chunks
            for column in chunk
        )

        return bytes


__all__ = ["StreamWindow", "WindowRows"]
