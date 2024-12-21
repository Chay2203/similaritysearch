from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import clip
from PIL import Image
import io
import requests
from enum import Enum
import base64

app = FastAPI(title="CLIP Embedding Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class InputType(str, Enum):
    text = "text"
    image_url = "image_url"
    image_base64 = "image_base64"

class EmbeddingRequest(BaseModel):
    type: InputType
    input: str

    class Config:
        json_schema_extra = {
            "example": {
                "type": "image_url",
                "input": "https://example.com/image.jpg"
            }
        }

async def download_image(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error downloading image: {str(e)}")

async def process_base64_image(base64_string: str) -> bytes:
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        return base64.b64decode(base64_string)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error processing base64 image: {str(e)}")

@app.post("/embeddings", response_model=dict)
async def get_embeddings(request: EmbeddingRequest):
    try:
        if request.type == InputType.text:
            text_input = clip.tokenize([request.input]).to(device)
            with torch.no_grad():
                embedding = model.encode_text(text_input)
                embedding = embedding.cpu().numpy().tolist()[0]
                print(f"Embedding length: {len(embedding)}")
            
        elif request.type == InputType.image_url:
            image_bytes = await download_image(request.input)
            image = Image.open(io.BytesIO(image_bytes))
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image_input)
                embedding = embedding.cpu().numpy().tolist()[0]
                print(f"Embedding length: {len(embedding)}")
                
        elif request.type == InputType.image_base64:
            # Process base64 encoded image
            image_bytes = await process_base64_image(request.input)
            image = Image.open(io.BytesIO(image_bytes))
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image_input)
                embedding = embedding.cpu().numpy().tolist()[0]
                print(f"Embedding length: {len(embedding)}")
        
        return {
            "status": "success",
            "embeddings": embedding,
            "embedding_dimension": len(embedding)
        }
        
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": device,
        "model": "ViT-B/32"
    }

@app.get("/examples")
async def get_examples():
    return {
        "text_example": {
            "type": "text",
            "input": "a photo of a cat"
        },
        "image_url_example": {
            "type": "image_url",
            "input": "https://example.com/cat.jpg"
        },
        "image_base64_example": {
            "type": "image_base64",
            "input": "base64_encoded_image_string"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)