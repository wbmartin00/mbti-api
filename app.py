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
    global _label_encoder
    if _label_encoder is None and LABEL_ENCODER_PATH.exists():
        with open(LABEL_ENCODER_PATH, "rb") as f:
            _label_encoder = pickle.load(f)
    return _label_encoder


def load_model():
    global _tokenizer, _model, _id2label

    if _model is not None:
        return

    import torch  # local import to keep startup light
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not MODEL_DIR.exists():
        raise RuntimeError(f"MODEL_DIR not found: {MODEL_DIR}")

    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    _model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    ).eval()

    # id -> label map
    le = load_label_encoder()
    if le is not None and hasattr(le, "classes_"):
        _id2label = {i: (lbl if isinstance(lbl, str) else str(lbl))
                     for i, lbl in enumerate(le.classes_)}
    else:
        _id2label = getattr(_model.config, "id2label", {}) or {}


class PredictIn(BaseModel):
    text: str


class PredictOut(BaseModel):
    label: str
    scores: List[float]

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

    # Defensive cap to avoid runaway costs
    text = text[:MAX_TEXT_CHARS]

    load_model()

    import torch
    with torch.no_grad():
        toks = _tokenizer(
            text,
            truncation=True,
            max_length=512,     # DistilBERT token limit
            return_tensors="pt",
        )
        logits = _model(**toks).logits[0]
        pred_id = int(torch.argmax(logits).item())
        scores = logits.tolist()

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores = probs.detach().cpu().numpy().flatten().tolist()

    label = _id2label.get(pred_id, f"LABEL_{pred_id}")
    return {"label": label, "prediction": label, "scores": scores}
