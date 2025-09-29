from __future__ import annotations

import asyncio
from typing import Any

import torch
import whisperx

from app.config import get_settings
from app.models import Segment, WordTiming


class WhisperXTranscriber:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = torch.device(self.settings.transcribe_device)
        self.model = whisperx.load_model(
            self.settings.whisper_model_size,
            device=self.device,
            compute_type=self.settings.whisper_compute_type,
        )
        self.alignment_model = None
        self.alignment_metadata = None
        self.diarization_pipeline = None

        if self.settings.enable_diarization and not self.settings.hf_token:
            raise RuntimeError("HF_TOKEN must be set when diarization is enabled")

    async def _ensure_alignment_model(self, language_code: str) -> None:
        if self.alignment_model is not None and self.alignment_metadata is not None:
            if self.alignment_metadata.get("language", {}).get("code") == language_code:
                return

        loop = asyncio.get_running_loop()
        self.alignment_model, self.alignment_metadata = await loop.run_in_executor(
            None,
            whisperx.load_align_model,
            language_code,
            self.device,
        )

    async def _ensure_diarization_pipeline(self) -> None:
        if not self.settings.enable_diarization:
            return
        if self.diarization_pipeline is not None:
            return
        loop = asyncio.get_running_loop()
        self.diarization_pipeline = await loop.run_in_executor(
            None,
            whisperx.DiarizationPipeline,
            self.settings.hf_token,
            self.device,
        )

    async def transcribe(self, media_path: str) -> tuple[list[Segment], dict[str, Any]]:
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(None, self.model.transcribe, media_path)
        language = result.get("language", "en")

        await self._ensure_alignment_model(language)

        aligned_segments = await loop.run_in_executor(
            None,
            whisperx.align,
            result["segments"],
            self.alignment_model,
            self.alignment_metadata,
            media_path,
            self.device,
        )

        segments = aligned_segments["segments"]

        if self.settings.enable_diarization:
            await self._ensure_diarization_pipeline()
            diarization_result = await loop.run_in_executor(
                None, self.diarization_pipeline, media_path
            )
            segments = whisperx.assign_word_speakers(diarization_result, segments)["segments"]

        parsed_segments: list[Segment] = []
        for seg in segments:
            words = [
                WordTiming(
                    start=word.get("start", seg["start"]),
                    end=word.get("end", seg["end"]),
                    text=word.get("word", ""),
                    speaker=str(word.get("speaker", seg.get("speaker", "UNKNOWN"))),
                )
                for word in seg.get("words", [])
            ]

            parsed_segments.append(
                Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text", ""),
                    speaker=str(seg.get("speaker", "UNKNOWN")),
                    words=words,
                )
            )

        return parsed_segments, result

