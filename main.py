from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI, APIError
import os
import uvicorn
import json

app = FastAPI(title="NVIDIA Proxy for JanitorAI")

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================================
# NVIDIA CLIENT
# ============================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY environment variable is not set")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    model: str
    messages: list

    temperature: float = 1.0
    max_tokens: int = 16384
    stream: bool = False

    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


# ============================================================
# MODELS
# ============================================================

@app.get("/v1/models")
async def list_models():
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
                "id": "deepseek-ai/deepseek-v4-flash",
                "object": "model",
                "created": 1700000000,
                "owned_by": "nvidia"
            },
            {
                "id": "deepseek-ai/deepseek-v4-pro",
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


# ============================================================
# OPTIONS / CORS
# ============================================================

@app.options("/v1/chat/completions")
@app.options("/{path:path}")
async def options_all(path: str = ""):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


# ============================================================
# CHAT COMPLETIONS
# ============================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):

    try:

        print("=" * 60)
        print(f"📥 Запрос модели: {request.model}")
        print(f"🌊 Streaming: {request.stream}")
        print(f"🧠 Max tokens: {request.max_tokens}")

        # ========================================================
        # ПАРАМЕТРЫ ЗАПРОСА
        # ========================================================

        params = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }

        if request.top_p != 1.0:
            params["top_p"] = request.top_p

        if request.frequency_penalty != 0.0:
            params["frequency_penalty"] = request.frequency_penalty

        if request.presence_penalty != 0.0:
            params["presence_penalty"] = request.presence_penalty

        # ========================================================
        # REASONING
        # ========================================================

        model_lower = request.model.lower()

        if "glm-5.2" in model_lower:

            print("🧠 GLM-5.2: включаем reasoning")

            params["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": True
                },
                "reasoning_effort": "max"
            }

        elif "deepseek-v4-flash" in model_lower:

            print("🧠 DeepSeek V4 Flash: включаем reasoning")

            params["extra_body"] = {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "max"
                }
            }

        elif "deepseek-v4-pro" in model_lower:

            print("🧠 DeepSeek V4 Pro: включаем reasoning")

            params["reasoning_effort"] = "max"

            params["extra_body"] = {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "max"
                }
            }

        elif "deepseek-v4" in model_lower:

            print("🧠 DeepSeek V4: включаем reasoning")

            params["reasoning_effort"] = "max"

            params["extra_body"] = {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "max"
                }
            }

        # ========================================================
        # REQUEST TO NVIDIA
        # ========================================================

        print("🔄 Отправка запроса в NVIDIA...")

        try:

            completion = client.chat.completions.create(**params)

            print("✅ NVIDIA приняла запрос")

        except APIError as e:

            print(f"❌ NVIDIA API Error: {str(e)}")

            error_body = getattr(e, "body", None)

            if error_body:
                print(f"📄 NVIDIA error body: {error_body}")

            return JSONResponse(
                status_code=getattr(e, "status_code", None) or 500,
                content={
                    "error": {
                        "message": str(e),
                        "type": "api_error",
                        "raw_response": str(error_body)
                    }
                }
            )

        except Exception as e:

            print(f"❌ Ошибка при запросе NVIDIA: {str(e)}")

            import traceback
            traceback.print_exc()

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(e),
                        "type": "internal_error"
                    }
                }
            )

        # ========================================================
        # STREAMING
        #
        # ВАЖНО:
        # При stream=True completion является итератором.
        # Нельзя обращаться к completion.choices[0].message
        # до обработки этого итератора.
        # ========================================================

        if request.stream:

            print("🌊 Запущен streaming")

            async def generate():

                try:

                    for chunk in completion:

                        # Иногда API может прислать chunk без choices
                        if not hasattr(chunk, "choices"):
                            continue

                        if not chunk.choices:
                            continue

                        delta = getattr(
                            chunk.choices[0],
                            "delta",
                            None
                        )

                        if delta is None:
                            continue

                        response_data = {
                            "choices": [
                                {
                                    "delta": {}
                                }
                            ]
                        }

                        # ------------------------------------------------
                        # Обычный текст
                        # ------------------------------------------------

                        content = getattr(
                            delta,
                            "content",
                            None
                        )

                        if content:

                            response_data["choices"][0]["delta"][
                                "content"
                            ] = content

                            print(
                                content,
                                end="",
                                flush=True
                            )

                        # ------------------------------------------------
                        # Reasoning
                        # ------------------------------------------------

                        reasoning = getattr(
                            delta,
                            "reasoning_content",
                            None
                        )

                        if not reasoning:

                            reasoning = getattr(
                                delta,
                                "reasoning",
                                None
                            )

                        if reasoning:

                            response_data["choices"][0]["delta"][
                                "reasoning_content"
                            ] = reasoning

                            print(
                                f"\n🧠 [reasoning chunk] "
                                f"{len(reasoning)} chars",
                                flush=True
                            )

                        # ------------------------------------------------
                        # Отправляем SSE chunk
                        # ------------------------------------------------

                        yield (
                            "data: "
                            + json.dumps(
                                response_data,
                                ensure_ascii=False
                            )
                            + "\n\n"
                        )

                    print("\n✅ Streaming завершён")

                except Exception as e:

                    print(
                        f"\n❌ Ошибка во время streaming: {str(e)}"
                    )

                    import traceback
                    traceback.print_exc()

                    error_data = {
                        "error": {
                            "message": str(e),
                            "type": "stream_error"
                        }
                    }

                    yield (
                        "data: "
                        + json.dumps(
                            error_data,
                            ensure_ascii=False
                        )
                        + "\n\n"
                    )

                finally:

                    print("📤 Отправляем [DONE]")

                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        # ========================================================
        # NON-STREAMING
        # ========================================================

        print("📦 Получен обычный ответ")

        if not hasattr(completion, "choices"):

            print("❌ В ответе отсутствует choices")

            try:
                if hasattr(completion, "model_dump"):
                    raw = completion.model_dump()
                else:
                    raw = str(completion)

            except Exception:
                raw = str(completion)

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Invalid response: missing choices",
                        "type": "api_error",
                        "raw_response": raw
                    }
                }
            )

        if not completion.choices:

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Empty choices",
                        "type": "api_error"
                    }
                }
            )

        # ========================================================
        # MESSAGE
        # ========================================================

        choice = completion.choices[0]

        message = getattr(
            choice,
            "message",
            None
        )

        if message is None:

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Missing message",
                        "type": "api_error"
                    }
                }
            )

        content = getattr(
            message,
            "content",
            ""
        )

        reasoning = getattr(
            message,
            "reasoning_content",
            None
        )

        if not reasoning:

            reasoning = getattr(
                message,
                "reasoning",
                None
            )

        print(
            f"📝 Контент: {len(content or '')} символов"
        )

        if reasoning:

            print(
                f"🧠 Reasoning: {len(reasoning)} символов"
            )

        # ========================================================
        # RESPONSE
        # ========================================================

        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content or ""
                    },
                    "finish_reason": getattr(
                        choice,
                        "finish_reason",
                        "stop"
                    ),
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        if reasoning:

            response_data["choices"][0]["message"][
                "reasoning_content"
            ] = reasoning

        print("📤 Отправляем обычный ответ")

        return JSONResponse(response_data)

    # ============================================================
    # GLOBAL ERROR
    # ============================================================

    except Exception as e:

        print(f"❌ Критическая ошибка: {str(e)}")

        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "ok",
        "message": "NVIDIA Proxy for JanitorAI"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            8000
        )
    )

    print(
        f"🚀 Запуск NVIDIA Proxy на порту {port}"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
