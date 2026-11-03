import os
from fastapi import FastAPI, HTTPException, Header
from openai import OpenAI

app = FastAPI(title="L.E.X.O. AI")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"

SYSTEM_PROMPT = """
Sen L.E.X.O. adlı ciddi ve cevap odaklı bir yapay zeka asistanısın.
Kullanıcıya Efendim diye hitap et.
Her zaman Türkçe ve en fazla 3 cümleyle, tek satır halinde cevap ver.
Çift tırnak ve noktalı virgül kullanma.
Bugünün tarihini kullanıcı sormadığı sürece söyleme.
Vindelst, Roma'dan sonraki Kuzey İtalya toprakları,
İsviçre ve Dalmaçya topraklarını içeren,
başkenti Vincia olan bir ülkedir.
Sayıları rakamlarla yaz.
"""

@app.get("/")
def home():
    return {"status": "online", "name": "LEXO AI"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL}

@app.get("/chat")
def chat(
    message: str,
    history: str = "",
    x_bot_token: str = Header(None)
):
    if x_bot_token != os.environ["BOT_TOKEN"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                    + "\nÖnceki konuşmalar:\n"
                    + history
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

        return {
            "success": True,
            "reply": answer or "Efendim, yanıt oluşturulamadı."
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
