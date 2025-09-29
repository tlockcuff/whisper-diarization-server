from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.transcription_service import TranscriptionService


app = FastAPI(title="GPU Speech to Text Service")
service = TranscriptionService()


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcriptions")
async def transcriptions(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    temp_path = Path("/tmp") / f"upload-{uuid.uuid4()}-{file.filename}"
    async with aiofiles.open(temp_path, "wb") as buffer:
        while chunk := await file.read(1 << 20):  # 1 MiB chunks
            await buffer.write(chunk)

    transcription_iterator = await service.transcribe(temp_path)

    async def event_stream():
        try:
            async for payload in transcription_iterator:
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            if temp_path.exists():
                temp_path.unlink()

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    response.background = BackgroundTasks()
    return response

