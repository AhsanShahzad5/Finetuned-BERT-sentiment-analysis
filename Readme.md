# Movie Review Sentiment Analyzer

A simple sentiment-analysis web app that predicts whether a movie review is **positive** or **negative**. It uses FastAPI, PyTorch, and a fine-tuned BERT model hosted on Hugging Face.

## Features

- Simple browser interface for entering reviews
- REST API with sentiment and confidence scores
- Health-check endpoint
- Automated tests with Pytest
- Docker and Docker Compose support

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000` in your browser. The model is downloaded from Hugging Face the first time the app starts.

## Run with Docker

```bash
docker compose up --build
```

## API example

Send a `POST` request to `/predict/`:

```json
{
  "text": "I really loved this movie!"
}
```

Example response:

```json
{
  "sentiment": "positive",
  "confidence": 0.9987
}
```

## Screenshots

![bert-sentiment-analysis-app](image.png)

![bert-sentiment-analysis-app](image-1.png)

## Model

This project uses [`ahsanfolium/ai-intern-imdb-sentiment-bert`](https://huggingface.co/ahsanfolium/ai-intern-imdb-sentiment-bert),

**a BERT model fine-tuned for IMDb sentiment classification.**
