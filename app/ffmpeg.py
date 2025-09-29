import asyncio
import shlex
import tempfile
from pathlib import Path


async def extract_audio(input_path: Path, sample_rate: int = 16000) -> Path:
    output_fd, output_path = tempfile.mkstemp(suffix=".wav")
    Path(output_path).unlink(missing_ok=True)
    cmd = (
        f"ffmpeg -hide_banner -loglevel error -i {shlex.quote(str(input_path))} "
        f"-ac 1 -ar {sample_rate} -f wav {shlex.quote(output_path)}"
    )
    proc = await asyncio.create_subprocess_shell(cmd)
    await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg failed to convert input media")
    return Path(output_path)

