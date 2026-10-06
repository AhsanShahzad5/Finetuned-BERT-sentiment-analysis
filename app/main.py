import logging
import time

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="Sentiment Analysis API")


@app.get("/healthz")
def health():
    return {"succesfully running"}


# serve html

app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Serve index.html at root
@app.get("/")
async def serve_homepage():
    return FileResponse("app/static/index.html")


# -----------------------------


# Load Week 4 model from Hugging Face Hub
MODEL_NAME = "ahsanfolium/ai-intern-imdb-sentiment-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    sentiment: str
    confidence: float


@app.post("/predict/", response_model=PredictionOut)
async def predict(input_text: TextIn):
    # 1. Tokenize
    inputs = tokenizer(
        input_text.text, return_tensors="pt", truncation=True, padding=True
    ).to(device)

    # 2. Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. Softmax to get confidence
    probs = F.softmax(outputs.logits, dim=-1)
    confidence, pred_class = torch.max(probs, dim=-1)

    # 4. Map class index to label
    sentiment = "positive" if pred_class.item() == 1 else "negative"

    return {"sentiment": sentiment, "confidence": round(confidence.item(), 4)}


# middleware and logging


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(f"{request.method} {request.url} completed_in={process_time:.4f}s")  # noqa: LOG015
    return response
