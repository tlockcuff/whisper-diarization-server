import os
import subprocess
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="Whisper Diarization Server")

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(..., media_type="audio/*"),
    model: Optional[str] = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    # Ignore other OpenAI params for now
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Create temporary file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        contents = await file.read()
        tmp_file.write(contents)
        audio_path = tmp_file.name

    try:
        # Build command for diarization
        cmd = [
            "python", "/whisper-diarization/diarize.py",
            "-a", audio_path
        ]
        if language:
            cmd.extend(["--language", language])

        # Start subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        def generate_output():
            # Stream stdout lines
            while True:
                output_line = process.stdout.readline()
                if output_line == '' and process.poll() is not None:
                    break
                if output_line:
                    yield output_line

            # Check for errors
            stderr_output = process.stderr.read()
            if process.returncode != 0:
                error_msg = f"Process failed with return code {process.returncode}: {stderr_output}"
                yield f"Error: {error_msg}\n"
            process.stdout.close()
            process.stderr.close()

        # For now, stream as text/plain
        # In future, can format as SSE or JSON chunks if needed
        return StreamingResponse(
            generate_output(),
            media_type="text/plain",
            headers={"X-Accel-Buffering": "no"}  # Disable buffering if needed
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
