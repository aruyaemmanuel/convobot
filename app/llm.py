import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=os.getenv("LLM_API_KEY"), timeout=60)


def complete(messages: list[dict]) -> tuple[str, dict]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    usage = resp.usage.model_dump() if resp.usage else {}
    return resp.choices[0].message.content, usage