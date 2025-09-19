#!/usr/bin/env python3
"""
Script to preload models during Docker build
"""
import os
import sys
from pathlib import Path

def preload_whisper_model():
    """Preload Whisper model"""
    try:
        from faster_whisper import WhisperModel
        
        asr_model = os.getenv('ASR_MODEL', 'base')
        cache_dir = '/app/cache/whisper'
        
        print(f"🔄 Loading Whisper model: {asr_model}")
        model = WhisperModel(asr_model, download_root=cache_dir)
        print(f"✅ Loaded Whisper model: {asr_model}")
        
    except Exception as e:
        print(f"⚠️ Failed to load Whisper model: {e}")
        # Don't fail the build, just warn
        pass

def preload_diarization_model():
    """Preload diarization model"""
    try:
        from pyannote.audio import Pipeline
        
        diarization_model = os.getenv('DIARIZATION_MODEL', 'pyannote/speaker-diarization@2.1')
        hf_token = os.getenv('HF_TOKEN')
        
        if not hf_token:
            print("⚠️ No HF_TOKEN provided, skipping diarization model preload")
            return
            
        print(f"🔄 Loading diarization model: {diarization_model}")
        
        pipeline = Pipeline.from_pretrained(diarization_model, use_auth_token=hf_token)
        print(f"✅ Loaded diarization model: {diarization_model}")
        
    except Exception as e:
        print(f"⚠️ Failed to load diarization model: {e}")
        # Don't fail the build, just warn
        pass

if __name__ == "__main__":
    print("🚀 Preloading models...")
    preload_whisper_model()
    preload_diarization_model()
    print("🎉 Model preloading completed!")
