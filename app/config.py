from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    whisper_model_size: str = Field(default="large-v3", alias="WHISPER_MODEL_SIZE")
    transcribe_device: Literal["cpu", "cuda"] = Field(default="cuda", alias="TRANSCRIBE_DEVICE")
    whisper_compute_type: Literal["float16", "bfloat16", "int8_float16", "int8"] = Field(
        default="float16", alias="WHISPER_COMPUTE_TYPE"
    )
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    enable_diarization: bool = Field(default=True, alias="ENABLE_DIARIZATION")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()

