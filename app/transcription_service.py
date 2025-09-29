from __future__ import annotations

import asyncio
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from app.ffmpeg import extract_audio
from app.models import StreamingPayload
from app.pipeline.streaming import stream_segments
from app.pipeline.transcriber import WhisperXTranscriber


class TranscriptionService:
    def __init__(self) -> None:
        self.transcriber = WhisperXTranscriber()

    async def transcribe(self, upload_path: Path) -> StreamingPayload:
        pcm_path = await extract_audio(upload_path)
        try:
            segments, metadata = await self.transcriber.transcribe(str(pcm_path))

            async def iterator() -> AsyncIterator[dict[str, object]]:
                yield {"type": "metadata", "value": metadata}
                async for payload in stream_segments(segments):
                    yield payload
                yield {"type": "done"}

            return iterator()
        finally:
            if pcm_path.exists():
                await asyncio.get_running_loop().run_in_executor(None, pcm_path.unlink)

