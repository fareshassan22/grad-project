import time
from openai import OpenAI
from config import OPENROUTER_API_KEY

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def generate_reply(transcript: str, max_length: int = 200, retries: int = 3) -> str:
    """
    Generate an Egyptian Arabic reply using GPT-OSS-20B via OpenRouter.
    Tuned for natural, friendly Egyptian dialect (اللهجة المصرية).
    """
    transcript = transcript.strip()
    if not transcript:
        return "مش سامعك كويس، ممكن تقول تاني؟"

    # 🇪🇬 Egyptian Arabic system prompt
    system_prompt = (
        "إنت مساعد ذكي وبتتكلم باللهجة المصرية الطبيعية. "
        "رد على المستخدم بطريقة ودودة وسهلة من غير أي رموز أو زخارف. "
        "خلي كلامك بسيط زي ما الناس في مصر بتتكلم في الحياة اليومية."
    )

    for attempt in range(1, retries + 1):
        try:
            print(f"[GPT-OSS-20B] Generating Egyptian reply (attempt {attempt}) for: {transcript}")
            start_time = time.time()

            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript},
                ],
                max_tokens=max_length,
                temperature=0.7,  # slightly more creative
            )

            reply = ""
            if completion and completion.choices:
                message = completion.choices[0].message
                if message and getattr(message, "content", None):
                    reply = message.content.strip()

            elapsed = time.time() - start_time
            if reply:
                print(f"[GPT-OSS-20B] ✅ Egyptian reply generated in {elapsed:.2f}s: {reply}")
                return reply

            print("[GPT-OSS-20B] ⚠️ Empty reply, retrying...")
            time.sleep(1)

        except Exception as e:
            print(f"[GPT-OSS-20B] ❌ Error on attempt {attempt}: {e}")
            time.sleep(2)

    return "معليش، حصلت مشكلة وأنا برد."
