# mbti-api
FastAPI backend for serving MBTI personality predictions from a fine-tuned DistilBERT model.   This service powers the interactive [MBTI Detector UI](https://github.com/wbmartin00/mbti-personality-detector-ui).


---

##  Live API

Base URL: **[https://mbti-api.fly.dev](https://mbti-api.fly.dev/)**

**Endpoints:**

- `GET /healthz` → health check (returns `{"ok": true}`)
- `POST /predict` → classify input text

Example request:

```bash
curl -X POST https://mbti-detector-api.fly.dev/predict   -H "Content-Type: application/json"   -d '{"text":"I love planning and abstract problem-solving."}'
```

Example response:

```json
{
  "label": "INTJ"
}
```

---

##  Project Structure

```
.
├── app.py              # FastAPI app (predict + healthz endpoints)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build for Fly.io
├── fly.toml            # Fly.io deployment config
└── models/             # (Fly volume mount, downloaded on first boot)
```

---

##  How It Works

1. The UI (static site) sends user text to this API via `/predict`.
2. On first boot, the API downloads the fine-tuned DistilBERT weights from Hugging Face.
3. The model is cached into a Fly.io **volume** (`/models`) so it persists across restarts.
4. FastAPI serves predictions with CORS enabled for the UI domain.

---

##  Deployment

This API is designed for deployment on [Fly.io](https://fly.io):

- Persistent storage via volumes (`fly volumes create models ...`)
- App configuration in `fly.toml`
- Auto-scaling + HTTPS by default

---

##  License

MIT License. See [LICENSE](LICENSE) for details.


```mermaid
flowchart LR
    user["Recruiter / User"] -->|opens| ui["MBTI Detector UI<br/>(GitHub Pages / Vercel / Netlify)"]
    ui -->|POST /predict { text }| api["FastAPI API<br/>mbti-detector-api.fly.dev"]

    subgraph fly["Fly.io (region: iad)"]
        api -->|loads once| model["DistilBERT (Transformers)"]
        model --- vol[("(Fly Volume)<br/>/models")]
        vol <-->|first boot download| hf["Hugging Face<br/>model repo"]
    end

    api -->|CORS OK| ui
    api -->|GET /healthz| health["Health Check"]
