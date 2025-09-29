from __future__ import annotations

from typing import AsyncIterator

from app.models import Segment


async def stream_segments(segments: list[Segment]) -> AsyncIterator[dict[str, object]]:
    for segment in segments:
        yield {
            "type": "segment",
            "start": segment.start,
            "end": segment.end,
            "speaker": segment.speaker,
            "text": segment.text,
        }
        for word in segment.words:
            yield {
                "type": "word",
                "start": word.start,
                "end": word.end,
                "speaker": word.speaker,
                "text": word.text,
            }

