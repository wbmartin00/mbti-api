# mbti-api
FastAPI backend for serving MBTI personality predictions from a fine-tuned DistilBERT model.   This service powers the interactive [MBTI Detector UI](https://github.com/wbmartin00/mbti-personality-detector-ui).

```mermaid
flowchart LR
    user[Recruiter / User] -->|opens| ui[MBTI Detector UI (GitHub Pages / Vercel / Netlify)]
    ui -->|POST /predict { text }| api[FastAPI API mbti-detector-api.fly.dev]

    subgraph fly[Fly.io (region: iad)]
        api -->|loads once| model[DistilBERT (Transformers)]
        model --- vol[(Fly Volume /models)]
        vol <-->|first boot download| hf[Hugging Face model repo]
    end

    api -->|CORS OK| ui
    api -->|GET /healthz| health[Health Check]
