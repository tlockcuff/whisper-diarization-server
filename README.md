# Whisper Diarization Server

# Goal
An API that receives an audio file (.mp3, .wav, .m4a, .ogg, .webm) and returns a text transcript with speaker diarization.

# Requirements
- Good error handling and debugging logs
- Docker support
- Ubuntu 24 Server
- 2x Nvidia 5060Ti GPU's

# Docker
Use docker compose to build a python server (FastAPI) that will allow file uploads via the OpenAI API spec (v1/audio/transcriptions)
The resulting uploading will be passed to the below tool for speaker diarization.
The server will response stream the text transcript and the speaker diarization back to the client.

# Speaker Diarization Using OpenAI Whisper
https://github.com/MahmoudAshraf97/whisper-diarization
Requirements: FFMPEG, Cython (sudo apt update && sudo apt install cython3)
git clone https://github.com/MahmoudAshraf97/whisper-diarization.git
cd whisper-diarization
pip install -c constraints.txt -r requirements.txt
python diarize.py -a AUDIO_FILE_NAME
