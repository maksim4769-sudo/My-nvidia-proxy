from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import uvicorn

app = FastAPI()

# Разрешаем CORS для JanitorAI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация клиента NVIDIA
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")  # Берем ключ из переменных окружения
)

# Модель для запроса от JanitorAI
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 1.0
    max_tokens: int = 16384
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        # Преобразуем запрос JanitorAI в запрос к NVIDIA
        completion = client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        # Если стриминг - возвращаем поток
        if request.stream:
            return StreamingResponse(stream_generator(completion), media_type="text/event-stream")
        
        # Если не стриминг - возвращаем JSON
        return {
            "choices": [{
                "message": {
                    "content": completion.choices[0].message.content
                }
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_generator(completion):
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            yield f"data: {chunk.choices[0].delta.content}\n\n"
    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
