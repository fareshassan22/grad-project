# streamlit_app.py
import streamlit as st
from utils.stt import transcribe
from utils.tts import synthesize
from utils.wav2lip_utils import create_lipsync_video, record_audio as record_audio_wav2lip
from utils.generate_reply import generate_reply
import os

st.set_page_config(page_title="Arabic Voice Assistant", layout="wide")
st.title("🎤 Arabic Voice Assistant + Avatar Lipsync")


st.header("Step 1: Record or Upload Audio")

# Upload option
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav", "mp3", "m4a"])

# Record option
record_duration = st.slider("Record duration (seconds)", 2, 15, 5)
record_button = st.button("🎙️ Record Audio")

audio_path = None
if uploaded_file:
    audio_path = f"output/{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(audio_path)
elif record_button:
    audio_path = record_audio_wav2lip(duration=record_duration)
    st.audio(audio_path)


transcript = ""
if audio_path:
    with st.spinner("📝 Transcribing audio..."):
        transcript = transcribe(audio_path)
    st.text_area("Transcribed Text:", transcript, height=100)


reply_text = ""
if transcript:
    with st.spinner("💬 Generating reply..."):
        reply_text = generate_reply(transcript)
    st.text_area("Reply Text:", reply_text, height=100)


speech_path = ""
if reply_text:
    with st.spinner("🔊 Synthesizing speech..."):
        speech_path = synthesize(reply_text)
    st.audio(speech_path)



st.markdown("---")
st.markdown("App powered by **Whisper STT**, **MMS-TTS Arabic**, **OpenRouter GPT-OSS-20B**, and **Wav2Lip**.")
