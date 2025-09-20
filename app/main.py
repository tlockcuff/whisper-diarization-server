import os
import subprocess
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info(f"Received request for audio transcription: filename={file.filename}, size={file.size if file.size else 'unknown'}, language={language}")
    
    if not file.filename:
        logger.warning("No file provided in request")
        raise HTTPException(status_code=400, detail="No file provided")

    # Create temporary file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        contents = await file.read()
        tmp_file.write(contents)
        audio_path = tmp_file.name

    logger.info(f"Uploaded file saved to temporary path: {audio_path}")

    try:
        # Build command for diarization
        cmd = [
            "python3", "/whisper-diarization/diarize.py",
            "-a", audio_path
        ]
        if language:
            cmd.extend(["--language", language])

        logger.info(f"Executing diarization command: {' '.join(cmd)}")

        # Start subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        logger.info(f"Subprocess started with PID: {process.pid}")

        def generate_output():
            # Stream stdout lines
            while True:
                output_line = process.stdout.readline()
                if output_line == '' and process.poll() is not None:
                    break
                if output_line:
                    logger.debug(f"Streaming output: {output_line.strip()}")
                    yield output_line

            # Wait for process to finish and check for errors
            process.wait()
            stderr_output = process.stderr.read()
            process.stdout.close()
            process.stderr.close()

            if process.returncode != 0:
                error_msg = f"Process failed with return code {process.returncode}: {stderr_output}"
                logger.error(error_msg)
                yield f"Error: {error_msg}\n"
            else:
                logger.info("Diarization process completed successfully")

        # For now, stream as text/plain
        # In future, can format as SSE or JSON chunks if needed
        return StreamingResponse(
            generate_output(),
            media_type="text/plain",
            headers={"X-Accel-Buffering": "no"}  # Disable buffering if needed
        )

    except Exception as e:
        logger.error(f"Exception in transcribe_audio: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            logger.info(f"Cleaned up temporary file: {audio_path}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
