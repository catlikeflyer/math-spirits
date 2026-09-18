import os
import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from server.engine import ModelEngine, DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger("math_spirits.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Math Spirits Inference Engine",
    description="Educational SLM dynamic model inference server for Math Spirits",
    version="1.0.0",
)

# Enable CORS for local web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = ModelEngine()


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Text content of the message")


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of conversation turns")
    model: Optional[str] = Field(None, description="Optional target model ID")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1024, ge=1, le=4096)
    system_prompt: Optional[str] = Field(None, description="Override system prompt")
    stream: Optional[bool] = Field(True, description="Whether to stream via SSE")


class ModelSelectRequest(BaseModel):
    model_id: str = Field(..., description="Model ID to activate")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "app": "Math Spirits Inference Engine",
        "version": "1.0.0",
        "active_model": engine.active_model_id,
    }


@app.get("/api/models")
async def list_models():
    """Dynamic Model Discovery: List all discovered models and their status."""
    engine.discover_models()
    models_list = []
    
    for mid, meta in engine.registered_models.items():
        models_list.append({
            "id": mid,
            "name": meta.get("name", mid),
            "display_name": meta.get("display_name", meta.get("name", mid)),
            "base_model": meta.get("base_model", "Unknown"),
            "adapter_path": meta.get("adapter_path", ""),
            "quantization": meta.get("quantization", "Q4_K_M"),
            "parameters": meta.get("parameters", "1.5B"),
            "context_window": meta.get("context_window", 4096),
            "type": meta.get("type", "mlx"),
            "description": meta.get("description", ""),
            "system_prompt": meta.get("system_prompt", ""),
            "is_active": (mid == engine.active_model_id),
        })
        
    return {
        "models": models_list,
        "active_model_id": engine.active_model_id,
        "total": len(models_list),
    }


@app.post("/api/models/select")
async def select_model(payload: ModelSelectRequest):
    """Dynamic Model Switching: Hot-swap active model in memory."""
    try:
        model_meta = engine.select_model(payload.model_id)
        return {
            "status": "success",
            "message": f"Successfully activated model {payload.model_id}",
            "active_model": {
                "id": payload.model_id,
                "display_name": model_meta.get("display_name", payload.model_id),
                "quantization": model_meta.get("quantization", "Q4_K_M"),
                "parameters": model_meta.get("parameters", "1.5B"),
                "type": model_meta.get("type", "mlx"),
            }
        }
    except Exception as e:
        logger.error(f"Error switching to model '{payload.model_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/system/status")
async def system_status():
    """System & Hardware telemetry (RAM, Apple Silicon Metal memory)."""
    return engine.get_system_telemetry()


@app.post("/api/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Streaming Chat Completions endpoint using Server-Sent Events (SSE).
    Emits token deltas in real-time, followed by inference telemetry.
    """
    messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]

    def event_generator():
        try:
            for event in engine.stream_chat(
                messages=messages_dict,
                temperature=req.temperature or 0.7,
                max_tokens=req.max_tokens or 1024,
                model_id=req.model,
                system_prompt=req.system_prompt,
            ):
                payload_json = json.dumps(event, ensure_ascii=False)
                yield f"data: {payload_json}\n\n"
        except Exception as err:
            logger.error(f"Error during streaming generation: {err}", exc_info=True)
            error_event = json.dumps({"type": "error", "error": str(err)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


def start():
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Math Spirits server on http://{host}:{port}")
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start()
