import os
import subprocess

# Run diarization on a dummy audio to download models
# Create a silent audio if needed
subprocess.run(["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=stereo", "-t", "10", "-y", "/tmp/silent.wav"])
cmd = ["python", "/whisper-diarization/diarize.py", "-a", "/tmp/silent.wav"]
subprocess.run(cmd)
os.unlink("/tmp/silent.wav")

print("Models downloaded.")
