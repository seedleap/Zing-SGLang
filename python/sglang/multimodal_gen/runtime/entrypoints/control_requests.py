# SPDX-License-Identifier: Apache-2.0
"""Control-request protocol between the HTTP process and scheduler workers.

These types are cross-process IPC contracts, not utilities: the HTTP side
constructs them; scheduler_client fans weight and shutdown operations out
to every replica and routes realtime controls to the session's replica.
Each scheduler dispatches them through ``Scheduler.request_handlers`` or
its realtime queue. Keep this module import-light -- both
processes import it, and the HTTP process must not drag in torch-heavy
worker modules through it.
"""

from typing import Any, List, Optional, Union

import msgspec


class SetLoraReq(msgspec.Struct):
    lora_nickname: Union[str, List[str]]
    lora_path: Optional[Union[str, List[Optional[str]]]] = None
    target: Union[str, List[str]] = "all"
    strength: Union[float, List[float]] = 1.0
    merge_mode: Optional[str] = None
    lora_alpha: Optional[Union[int, List[Optional[int]]]] = None


class MergeLoraWeightsReq(msgspec.Struct):
    target: str = "all"
    strength: float = 1.0


class UnmergeLoraWeightsReq(msgspec.Struct):
    target: str = "all"


class ListLorasReq(msgspec.Struct):
    pass


class ShutdownReq(msgspec.Struct):
    pass


class ReleaseRealtimeSessionReq(msgspec.Struct):
    session_id: str


class GetDisaggStatsReq(msgspec.Struct):
    """Request to get disagg pipeline metrics from the scheduler."""

    pass


class ReplaceQueuedRealtimeReq(msgspec.Struct):
    session_id: str
    generation_id: str
    chunk_index: int
    request_id: str
    replacement: Any
