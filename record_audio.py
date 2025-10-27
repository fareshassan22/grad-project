import sounddevice as sd
import wavio

def record_audio(path="input.wav", duration=5, fs=16000):
    print("🎙️ Recording for", duration, "seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wavio.write(path, audio, fs, sampwidth=2)
    print("✅ Saved recording to", path)
    return path
