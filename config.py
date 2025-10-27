import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()



# OpenRouter (for GPT model)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ElevenLabs (for Speech-to-Text and Text-to-Speech)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("TTS_API_KEY")

# Default language (Arabic)
LANGUAGE = os.getenv("LANGUAGE", "ar")



if not OPENROUTER_API_KEY:
    raise EnvironmentError("❌ OPENROUTER_API_KEY not set. Please add it to your .env file or environment.")

if not ELEVENLABS_API_KEY:
    raise EnvironmentError("❌ ELEVENLABS_API_KEY not set. Please add it to your .env file or environment.")
