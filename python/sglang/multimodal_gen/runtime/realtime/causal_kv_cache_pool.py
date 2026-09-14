# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch


class RealtimeKVCachePoolCapacityError(RuntimeError):
    pass


@dataclass(slots=True)
class RealtimeKVCachePoolSlot:
    slot_id: int
    kv_cache: list[Any]
    crossattn_cache: list[Any]
    in_use: bool = False


class RealtimeKVCachePoolLease:
    """An idempotent lease for one statically allocated realtime cache slot."""

    def __init__(
        self,
        pool: RealtimeKVCachePool,
        slot: RealtimeKVCachePoolSlot,
        *,
        bucket_key: str | None = None,
    ) -> None:
        self._pool = pool
        self._slot: RealtimeKVCachePoolSlot | None = slot
        self.bucket_key = bucket_key

    @property
    def slot_id(self) -> int:
        if self._slot is None:
            raise RuntimeError("Realtime KV cache pool lease has been released")
        return self._slot.slot_id

    @property
    def kv_cache(self) -> list[Any]:
        if self._slot is None:
            raise RuntimeError("Realtime KV cache pool lease has been released")
        return self._slot.kv_cache

    @property
    def crossattn_cache(self) -> list[Any]:
        if self._slot is None:
            raise RuntimeError("Realtime KV cache pool lease has been released")
        return self._slot.crossattn_cache

    def release(self) -> None:
        slot = self._slot
        if slot is None:
            return
        self._slot = None
        self._pool._release(slot)


class RealtimeKVCachePool:
    """Fixed-size cache pool for one serialized GPU worker compute stream."""

    def __init__(self, slots: list[RealtimeKVCachePoolSlot]) -> None:
        if not slots:
            raise ValueError("Realtime KV cache pool requires at least one slot")
        self._slots = slots
        self._available = deque(slots)
        self.acquire_count = 0
        self.release_count = 0

    @property
    def size(self) -> int:
        return len(self._slots)

    @property
    def available(self) -> int:
        return len(self._available)

    def acquire(self, *, bucket_key: str | None = None) -> RealtimeKVCachePoolLease:
        if not self._available:
            raise RealtimeKVCachePoolCapacityError(
                "Realtime KV cache pool capacity exhausted: "
                f"active={self.size} size={self.size}"
            )
        slot = self._available.popleft()
        slot.in_use = True
        self.acquire_count += 1
        return RealtimeKVCachePoolLease(self, slot, bucket_key=bucket_key)

    def _release(self, slot: RealtimeKVCachePoolSlot) -> None:
        if not slot.in_use:
            return
        slot.in_use = False
        self._available.append(slot)
        self.release_count += 1


def allocated_cache_bytes(slots: list[RealtimeKVCachePoolSlot]) -> int:
    """Count unique tensor storage referenced by the preallocated slots."""

    seen: set[tuple[str, int]] = set()
    total = 0
    for slot in slots:
        for cache in (*slot.kv_cache, *slot.crossattn_cache):
            if not is_dataclass(cache):
                continue
            for cache_field in fields(cache):
                value = getattr(cache, cache_field.name)
                if not isinstance(value, torch.Tensor):
                    continue
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr())
                if key in seen:
                    continue
                seen.add(key)
                total += storage.nbytes()
    return total
