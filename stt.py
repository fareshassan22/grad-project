#now uses ElevenLabs STT instead of Whisper

import os
import librosa
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

# ✅ Load environment variables from .env
load_dotenv()

# Optional noise reduction
try:
    import noisereduce as nr
    NOISE_REDUCTION = True
except ImportError:
    NOISE_REDUCTION = False

from elevenlabs import ElevenLabs

# ==========================
# Initialize ElevenLabs Client
# ==========================
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVEN_API_KEY:
    raise EnvironmentError(
        "❌ ELEVENLABS_API_KEY not set. Please run: "
        "set ELEVENLABS_API_KEY=your_api_key_here on Windows or "
        "export ELEVENLABS_API_KEY=your_api_key_here on macOS/Linux."
    )

client = ElevenLabs(api_key=ELEVEN_API_KEY)
print("[ElevenLabs STT] ✅ Initialized client.")

# ==========================
# Audio Preprocessing
# ==========================
def preprocess_audio(audio_path: str, target_sr: int = 16000) -> str:
    """Load audio, resample, normalize, optionally denoise, and save a WAV file."""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    if len(audio) == 0:
        raise ValueError(f"Audio is empty: {audio_path}")

    # Normalize audio
    audio = audio / np.max(np.abs(audio))

    # Optional noise reduction
    if NOISE_REDUCTION:
        print("[ElevenLabs STT] Applying noise reduction...")
        audio = nr.reduce_noise(y=audio, sr=sr)

    # Save preprocessed audio
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    temp_path = os.path.join(output_dir, f"preprocessed_{os.path.basename(audio_path)}")
    sf.write(temp_path, audio, target_sr, subtype="PCM_16")
    print(f"[ElevenLabs STT] Preprocessed audio saved to: {temp_path}")
    return temp_path

# ==========================
# Transcription Function
# ==========================
def transcribe(audio_path: str, language: str = "ar") -> str:
    """Transcribe Arabic (or multilingual) audio using ElevenLabs STT (scribe_v1)."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    preprocessed_path = preprocess_audio(audio_path)
    print(f"[ElevenLabs STT] Transcribing: {preprocessed_path} ...")

    try:
        with open(preprocessed_path, "rb") as audio_file:
            # ✅ Use the correct ElevenLabs STT model
            response = client.speech_to_text.convert(
                model_id="scribe_v1",  # valid model_id
                file=audio_file,
            )

        # Extract text
        text = ""
        if isinstance(response, dict) and "text" in response:
            text = response["text"].strip()
        elif hasattr(response, "text"):
            text = response.text.strip()

        if text:
            print(f"[ElevenLabs STT] ✅ Transcription complete: {text}")
        else:
            print("[ElevenLabs STT] ⚠ No text recognized.")
            text = ""

        return text

    except Exception as e:
        print(f"[ElevenLabs STT] ❌ Error during transcription: {e}")
        return ""