import os
import time
from elevenlabs import ElevenLabs
from elevenlabs.core import ApiError  # ✅ Correct import for API errors

# ==============================
# 🔑 Load ElevenLabs API Key
# ==============================
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVEN_API_KEY:
    raise EnvironmentError("❌ ELEVENLABS_API_KEY not set. Please export it first.")

# ==============================
# 📁 Output Directory
# ==============================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 🤖 Initialize Client
# ==============================
client = ElevenLabs(api_key=ELEVEN_API_KEY)
print("[TTS] ✅ ElevenLabs client initialized.")


# ==============================
# 🎙️ Egyptian Arabic TTS
# ==============================
def synthesize(
    text: str,
    out_path: str = None,
    voice_id: str = "wxweiHvoC2r2jFM7mS8b",  # Your custom voice ID
    dialect: str = "egyptian",
    retries: int = 3
) -> str:
    """
    Convert text to speech using ElevenLabs multilingual TTS.
    Supports Egyptian Arabic (اللَّهجة المصرية) by conditioning the text.
    """

    if not text.strip():
        raise ValueError("❌ Empty text for TTS.")

    # Output path setup
    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, "reply.wav")
    if not out_path.lower().endswith(".wav"):
        out_path += ".wav"

    # 🇪🇬 Dialect conditioning for Egyptian Arabic
    if dialect.lower() == "egyptian":
        conditioned_text = f" {text.strip()}"
    else:
        conditioned_text = text.strip()

    print(f"[TTS] 🎧 Synthesizing in {dialect} Arabic using voice_id={voice_id}...")

    for attempt in range(retries):
        try:
            # Generate audio stream
            audio_stream = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",  # Stable multilingual TTS model
                text=conditioned_text,
            )

            # Write to file
            with open(out_path, "wb") as f:
                for chunk in audio_stream:
                    f.write(chunk)

            print(f"[TTS] ✅ Egyptian Arabic audio saved: {out_path}")
            return out_path

        except ApiError as e:
            if "system_busy" in str(e) and attempt < retries - 1:
                print(f"[TTS] ⚠️ System busy, retrying ({attempt+1}/{retries})...")
                time.sleep(2)
                continue
            else:
                print(f"[TTS] ❌ ElevenLabs API error: {e}")
                raise

        except Exception as e:
            print(f"[TTS] ❌ Unexpected error: {e}")
            raise
