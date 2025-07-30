from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
import os
import platform
from typing import Optional, List

# Initialize the model from local file
llm = Llama(
    model_path="./mistral-7b-instruct-v0.1.Q5_K_S.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

app = FastAPI(title="Llama Server", version="1.0.0")

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    echo: bool = False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95

@app.get("/", response_class=HTMLResponse)
async def serve_chatbot():
    try:
        with open("chatbot.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Chatbot not found</h1><p>Please ensure chatbot.html exists in the same directory.</p>")

@app.get("/health")
async def health():
    return {"message": "Llama Server is running"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "mistral-7b-instruct",
                "object": "model",
                "created": 1677610602,
                "owned_by": "local"
            }
        ]
    }

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    try:
        output = llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=request.echo
        )
        return {
            "id": "cmpl-local",
            "object": "text_completion",
            "model": "mistral-7b-instruct",
            "choices": [
                {
                    "text": output["choices"][0]["text"],
                    "index": 0,
                    "finish_reason": output["choices"][0]["finish_reason"]
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Convert chat messages to a single prompt
        prompt = ""
        for message in request.messages:
            if message.role == "system":
                prompt += f"System: {message.content}\n"
            elif message.role == "user":
                prompt += f"User: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"
        prompt += "Assistant: "
        
        output = llm(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["User:", "System:"]
        )
        
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "model": "mistral-7b-instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output["choices"][0]["text"].strip()
                    },
                    "finish_reason": output["choices"][0]["finish_reason"]
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)