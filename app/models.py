from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator


@dataclass(slots=True)
class WordTiming:
    start: float
    end: float
    text: str
    speaker: str


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str
    words: list[WordTiming]


StreamingPayload = AsyncIterator[dict[str, object]]

