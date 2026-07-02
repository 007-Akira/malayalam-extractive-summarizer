from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from summarize import DEFAULT_MODEL_KEY, summarize_article


app = FastAPI(title="Malayalam Extractive Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    sentence_count: int = Field(3, ge=1, le=10)
    diversity: float | Literal["auto"] = "auto"
    model: Literal[
        "sentence_classifier",
        "hybrid_classifier",
        "muril_classifier",
        "chotta_bheem",
        "chotta_bheem_v2",
    ] = DEFAULT_MODEL_KEY


class SummarizeResponse(BaseModel):
    summary: str
    sentences: list[str]
    sentence_count: int
    diversity: float | Literal["auto"]
    model: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    try:
        summary, sentences = summarize_article(
            request.text,
            k=request.sentence_count,
            diversity=request.diversity,
            model_key=request.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if "Article is too short" in summary:
        raise HTTPException(status_code=422, detail=summary)

    return SummarizeResponse(
        summary=summary,
        sentences=sentences,
        sentence_count=request.sentence_count,
        diversity=request.diversity,
        model=request.model,
    )
