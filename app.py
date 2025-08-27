# app.py
import os
import pathlib
import pickle
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------- CONFIG -----------------
ALLOWED_ORIGINS = [
    "https://wbmartin00.github.io"
]

MODEL_DIR = pathlib.Path(os.getenv("MODEL_DIR", "/models/mbti-distilbert-model"))
LABEL_ENCODER_PATH = pathlib.Path(os.getenv("LABEL_ENCODER_PATH", "/app/label_encoder.pkl"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "200000"))
# ------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Lazy singletons
_tokenizer = None
_model = None
_id2label = None
_label_encoder = None


def load_label_encoder():
    """Try to load sklearn LabelEncoder if available."""
    global _label_encoder
    if _label_encoder is None and LABEL_ENCODER_PATH.exists():
        with open(LABEL_ENCODER_PATH, "rb") as f:
            _label_encoder = pickle.load(f)
    return _label_encoder


def build_id2label():
    """Construct id->label mapping with safe fallback chain."""
    global _id2label

    # 1. Try sklearn LabelEncoder (preferred)
    le = load_label_encoder()
    if le is not None and hasattr(le, "classes_"):
        _id2label = {i: (cls if isinstance(cls, str) else str(cls))
                     for i, cls in enumerate(le.classes_)}
        return

    # 2. Try huggingface model config
    if getattr(_model.config, "id2label", None):
        _id2label = {int(i): str(lbl) for i, lbl in _model.config.id2label.items()}
        return

    # 3. Hardcoded MBTI fallback (alphabetical order like LabelEncoder)
    mbti_alpha = [
        "ENFJ","ENFP","ENTJ","ENTP","ESFJ","ESFP","ESTJ","ESTP",
        "INFJ","INFP","INTJ","INTP","ISFJ","ISFP","ISTJ","ISTP"
    ]
    _id2label = {i: t for i, t in enumerate(mbti_alpha)}


def load_model():
    global _tokenizer, _model, _id2label

    if _model is not None:
        return

    import torch
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not MODEL_DIR.exists():
        raise RuntimeError(f"MODEL_DIR not found: {MODEL_DIR}")

    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    _model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    ).eval()

    build_id2label()


class PredictIn(BaseModel):
    text: str


class PredictOut(BaseModel):
    scores: List[float]
    prediction: str


@app.get("/")
def root():
    return {"ok": True, "service": "mbti-api"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    text = (inp.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    # Defensive cap
    text = text[:MAX_TEXT_CHARS]

    load_model()

    import torch
    with torch.no_grad():
        toks = _tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        logits = _model(**toks).logits[0]
        pred_id = int(torch.argmax(logits).item())
        scores = logits.tolist()

    return {
        "scores": scores,
        "prediction": _id2label.get(pred_id, str(pred_id))
    }