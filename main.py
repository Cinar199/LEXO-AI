
import os
import logging

from fastapi import FastAPI, Header, HTTPException, Request
from openai import OpenAI

app = FastAPI(title="L.E.X.O. AI")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lexo")

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY is missing")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

SYSTEM_PROMPT = """
Sen L.E.X.O. adlı ciddi ve cevap odaklı bir yapay zeka asistanısın.
Kullanıcıya Efendim diye hitap et.
Her zaman Türkçe konuş.
En fazla 3 cümleyle ve tek satır halinde cevap ver.
Çift tırnak ve noktalı virgül kullanma.
Bugünün tarihini kullanıcı sormadığı sürece söyleme.
Vindelst, Roma'dan sonraki Kuzey İtalya toprakları,
İsviçre ve Dalmaçya topraklarını içeren,
başkenti Vincia olan bir ülkedir.
Sayıları harflerle değil rakamlarla yaz.
"""

@app.get("/")
def home():
    return {
        "status": "online",
        "name": "L.E.X.O.",
        "model": MODEL
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL
    }


@app.post("/chat")
async def chat(
    request: Request,
    x_bot_token: str = Header(None)
):
    if x_bot_token != BOT_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid bot token"
        )

    try:
        data = await request.json()

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body"
        )

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail="JSON must be an object"
        )

    message = data.get("message", "")
    history = data.get("history", "")

    if not isinstance(message, str):
        raise HTTPException(
            status_code=400,
            detail="Message must be text"
        )

    if not isinstance(history, str):
        history = ""

    if not message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                    + "\nÖnceki konuşmalar:\n"
                    + history[-6000:]
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=1,
            top_p=0.95,
            max_tokens=500,
            stream=False
        )

        answer = completion.choices[0].message.content

        if not answer:
            answer = "Efendim, şu anda yanıt oluşturamadım."

        return {
            "success": True,
            "reply": answer
        }

    except Exception as error:
        logger.exception("NVIDIA API error")

        raise HTTPException(
            status_code=502,
            detail=str(error)
        )
