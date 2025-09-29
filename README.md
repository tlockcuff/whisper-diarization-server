# Whisper Diarization Server

GPU-accelerated speech-to-text and diarization microservice that streams transcription results over SSE. Designed for NVIDIA RTX 5060 Ti (compute capability 12.0) by compiling PyTorch from source with CUDA 12.4 support.

## Features

- Upload audio/video (any FFmpeg-supported format)
- WhisperX transcription with word-level timestamps
- Pyannote diarization (requires Hugging Face token)
- Streaming responses in OpenAI-style Server-Sent Events
- Stateless processing: files are deleted after transcription

## Prerequisites

- Docker 24+
- NVIDIA drivers + NVIDIA Container Toolkit
- Hugging Face access token with rights to `pyannote` pipelines

## Build & Run

- `docker compose build`
- `docker compose up`

API available at `http://localhost:9001`.

### Environment Variables

Set these in your shell or `.env` file:

- `HF_TOKEN`: Hugging Face token for diarization models
- `TRANSCRIBE_DEVICE`: `cuda` or `cpu`
- `WHISPER_MODEL_SIZE`: defaults to `large-v3`
- `WHISPER_COMPUTE_TYPE`: e.g., `float16`

## API

### `POST /transcriptions`

Multipart upload with `file` field. Response streams SSE events:

- `curl -N -X POST http://localhost:9001/transcriptions -H "Accept: text/event-stream" -F "file=@sample.wav"`

Event payload examples:

```json
{"type":"metadata","value":{"language":"en","duration":12.5}}
{"type":"segment","start":0.4,"end":3.2,"speaker":"SPEAKER_00","text":"Hello there"}
{"type":"word","start":0.4,"end":0.8,"speaker":"SPEAKER_00","text":"Hello"}
{"type":"done"}
```

## Implementation Notes

- PyTorch and torchaudio are built in a separate stage targeting `sm_120`.
- FFmpeg converts uploads to 16 kHz mono WAV before transcription.
- WhisperX loads diarization models only when `HF_TOKEN` is present.

## Troubleshooting

- GPU errors: ensure host driver >= CUDA 12.4 and `docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi` works.
- Diarization fails: verify `HF_TOKEN` and network access to Hugging Face.
- Slow builds: first build compiles PyTorch; subsequent builds reuse cached wheels.
