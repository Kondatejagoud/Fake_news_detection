import os
import json
import wave
import subprocess
import imageio_ffmpeg
from vosk import Model, KaldiRecognizer
from app.core.config import settings
from app.core.logging import logger

_vosk_model = None

def get_vosk_model():
    global _vosk_model
    if _vosk_model is None:
        model_path = settings.VOSK_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk offline model not found at path: {model_path}")
        logger.info(f"Loading Vosk model from: {model_path}...")
        _vosk_model = Model(model_path)
        logger.info("Vosk model successfully loaded.")
    return _vosk_model

def extract_audio_to_wav(video_path: str, output_wav_path: str) -> bool:
    """
    Extracts the audio channel from a video file and transcode it to 16kHz mono 16-bit PCM WAV.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # -vn: skip video stream
    # -acodec pcm_s16le: write 16-bit little-endian PCM samples
    # -ar 16000: resample to 16000Hz sampling rate
    # -ac 1: set mono channel
    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", output_wav_path
    ]
    
    try:
        logger.info(f"Extracting WAV audio stream from {video_path} to {output_wav_path}...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception as e:
        logger.error(f"Failed to extract audio track: {e}")
        return False

def transcribe_audio_to_text(wav_path: str) -> str:
    """
    Reads a 16kHz PCM WAV file and transcribes it offline using Kaldi speech recognizer.
    """
    if not os.path.exists(wav_path):
        logger.warning(f"Audio file missing for transcription: {wav_path}")
        return ""
        
    try:
        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            logger.error("PCM WAV validation failed: Audio must be mono 16-bit PCM WAV.")
            wf.close()
            return ""
            
        model = get_vosk_model()
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(False)
        
        transcripts = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res_dict = json.loads(rec.Result())
                text = res_dict.get("text", "")
                if text:
                    transcripts.append(text)
                    
        final_dict = json.loads(rec.FinalResult())
        final_text = final_dict.get("text", "")
        if final_text:
            transcripts.append(final_text)
            
        wf.close()
        full_transcript = " ".join(transcripts).strip()
        logger.info(f"Vosk speech-to-text transcription complete. Length: {len(full_transcript)} chars.")
        return full_transcript
    except Exception as e:
        logger.error(f"Vosk transcription pipeline failed: {e}")
        return ""
