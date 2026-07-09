from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import uvicorn
import json
import sys

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Инициализация клиента NVIDIA
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY environment variable is not set")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Модель для запроса от JanitorAI
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 1.0
    max_tokens: int = 16384
    stream: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

# 🔥 НОВЫЙ ЭНДПОИНТ ДЛЯ CHUB.AI
@app.get("/v1/models")
async def list_models():
    """Возвращает список доступных моделей для Chub.ai"""
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "z-ai/glm-5.2",
                "object": "model",
                "created": 1700000000,
                "owned_by": "nvidia"
            },
            {
                "id": "meta/llama-3.1-8b-instruct",
                "object": "model",
                "created": 1700000000,
                "owned_by": "nvidia"
            },
            {
                "id": "meta/llama-3.1-70b-instruct",
                "object": "model",
                "created": 1700000000,
                "owned_by": "nvidia"
            },
            {
                "id": "nvidia/nemotron-mini-4b-instruct",
                "object": "model",
                "created": 1700000000,
                "owned_by": "nvidia"
            }
        ]
    })

@app.options("/v1/chat/completions")
async def options_chat():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.options("/{path:path}")
async def options_all(path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        print("=" * 50)
        print(f"📥 Получен запрос для модели: {request.model}")
        
        # Базовые параметры для NVIDIA
        params = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }
        
        # Добавляем опциональные параметры
        if request.top_p != 1.0:
            params["top_p"] = request.top_p
        if request.frequency_penalty != 0.0:
            params["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty != 0.0:
            params["presence_penalty"] = request.presence_penalty
        
        # 🔥 МАКСИМАЛЬНЫЙ REASONING ДЛЯ GLM-5.2
        if "glm-5.2" in request.model.lower():
            print("🧠 Активация МАКСИМАЛЬНОГО reasoning для GLM-5.2")
            params["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": True
                },
                "reasoning_effort": "max"
            }
        
        print("🔄 Отправка запроса в NVIDIA...")
        completion = client.chat.completions.create(**params)
        print("✅ Ответ от NVIDIA получен")
        
        # Безопасное извлечение данных
        if not hasattr(completion, 'choices') or not completion.choices:
            print("❌ Нет поля choices в ответе")
            raise ValueError("Invalid response format: missing choices")
        
        choice = completion.choices[0]
        message = getattr(choice, 'message', None)
        if not message:
            print("❌ Нет поля message в ответе")
            raise ValueError("Invalid response format: missing message")
        
        # Извлекаем content и reasoning_content
        content = getattr(message, 'content', '')
        reasoning_content = getattr(message, 'reasoning_content', None)
        
        print(f"📝 Длина контента: {len(content)} символов")
        if reasoning_content:
            print(f"🧠 Reasoning найден! Длина: {len(reasoning_content)} символов")
        else:
            print("❌ Reasoning ОТСУТСТВУЕТ в ответе NVIDIA")
        
        # Обработка стриминга
        if request.stream:
            def generate():
                try:
                    for chunk in completion:
                        if not hasattr(chunk, 'choices') or not chunk.choices:
                            continue
                        delta = getattr(chunk.choices[0], 'delta', None)
                        if not delta:
                            continue
                        
                        response_data = {"choices": [{"delta": {}}]}
                        
                        delta_content = getattr(delta, 'content', None)
                        if delta_content:
                            response_data["choices"][0]["delta"]["content"] = delta_content
                        
                        delta_reasoning = getattr(delta, 'reasoning_content', None)
                        if delta_reasoning:
                            print("🧠 Reasoning в стриминге")
                            response_data["choices"][0]["delta"]["reasoning_content"] = delta_reasoning
                        
                        yield f"data: {json.dumps(response_data)}\n\n"
                except Exception as e:
                    print(f"❌ Ошибка в стриминге: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                finally:
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        # Обычный ответ
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        if reasoning_content:
            response_data["choices"][0]["message"]["reasoning_content"] = reasoning_content
        
        print("📤 Отправка ответа клиенту")
        return JSONResponse(response_data)
        
    except Exception as e:
        print(f"❌ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "api_error"
                }
            }
        )

@app.get("/")
def root():
    return {"status": "ok", "message": "NVIDIA Proxy for JanitorAI"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Запуск сервера на порту {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
