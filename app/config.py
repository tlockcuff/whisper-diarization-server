"""Configuration management for Whisper Diarization Server."""

import os
from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AudioConfig(BaseModel):
    """Audio processing configuration."""

    max_length_seconds: int = Field(default=300, description="Maximum audio length in seconds")
    supported_formats: List[str] = Field(
        default=["mp3", "wav", "flac", "m4a", "ogg", "webm"],
        description="Supported audio formats"
    )
    sample_rate: int = Field(default=16000, description="Target sample rate for audio processing")
    batch_size: int = Field(default=8, description="Batch size for processing")


class ModelConfig(BaseModel):
    """Model configuration."""

    whisper_model: str = Field(default="large-v2", description="Whisper model name")
    pyannote_model: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="Pyannote diarization model"
    )
    cache_dir: Path = Field(default=Path("./cache"), description="Model cache directory")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")


class HardwareConfig(BaseModel):
    """Hardware configuration."""

    use_gpu: bool = Field(default=True, description="Use GPU acceleration if available")
    gpu_device: int = Field(default=0, description="GPU device index")
    max_workers: int = Field(default=2, description="Maximum number of worker processes")


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="json", description="Log format (json or text)")


class SecurityConfig(BaseModel):
    """Security configuration."""

    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins"
    )
    api_key_required: bool = Field(default=False, description="Require API key authentication")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    class Config:
        env_file = ".env"
        case_sensitive = False

    # Core configurations
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    hardware: HardwareConfig = HardwareConfig()
    audio: AudioConfig = AudioConfig()
    logging: LoggingConfig = LoggingConfig()
    security: SecurityConfig = SecurityConfig()

    # External services
    huggingface_token: Optional[str] = Field(default=None, description="Hugging Face API token")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")
    enable_redis: bool = Field(default=False, description="Enable Redis caching")

    # Feature flags
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


# Global settings instance
settings = get_settings()