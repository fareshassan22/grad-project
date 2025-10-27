import subprocess
import os
import time
import sounddevice as sd
import soundfile as sf

# Paths
WAV2LIP_PATH = "Wav2Lip"  # Path to cloned Wav2Lip repo
WAV2LIP_INFERENCE = os.path.join(WAV2LIP_PATH, "inference.py")
WAV2LIP_MODEL = os.path.join(WAV2LIP_PATH, "checkpoints", "wav2lip_gan.pth")  # user must download

def create_lipsync_video(avatar_image, audio_wav, out_video="output/avatar_lipsync.mp4", fps=25):
    """
    Creates a lip-synced video from avatar_image + audio_wav using Wav2Lip.
    Requires: clone https://github.com/Rudrabha/Wav2Lip and download checkpoints/wav2lip_gan.pth
    """
    os.makedirs(os.path.dirname(out_video), exist_ok=True)

    # Check if files exist
    if not os.path.isfile(avatar_image):
        raise FileNotFoundError(f"Avatar image not found: {avatar_image}")
    if not os.path.isfile(audio_wav):
        raise FileNotFoundError(f"Audio file not found: {audio_wav}")
    if not os.path.isfile(WAV2LIP_MODEL):
        raise FileNotFoundError(f"Wav2Lip model not found: {WAV2LIP_MODEL}")

    cmd = [
        "python", WAV2LIP_INFERENCE,
        "--checkpoint_path", WAV2LIP_MODEL,
        "--face", avatar_image,
        "--audio", audio_wav,
        "--outfile", out_video,
        "--fps", str(fps)
    ]
    print("[wav2lip_utils] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[wav2lip_utils] Video saved to {out_video}")
    return out_video

def record_audio(duration=5, out_path=None, samplerate=16000, channels=1):
    """
    Records audio from the microphone and saves it to out_path.
    """
    if out_path is None:
        out_path = f"output/input_{int(time.time())}.wav"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        print(f"[wav2lip_utils] Recording audio for {duration} seconds...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
        sd.wait()
        sf.write(out_path, audio, samplerate)
        print(f"[wav2lip_utils] Saved audio to {out_path}")
        return out_path
    except Exception as e:
        print("[wav2lip_utils] Error recording audio:", e)
        return None
