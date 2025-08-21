# mbti-api
FastAPI backend for serving MBTI personality predictions from a fine-tuned DistilBERT model.   This service powers the interactive [MBTI Detector UI](https://github.com/wbmartin00/mbti-personality-detector-ui).

flowchart LR
  user[Recruiter / User] -->|opens| ui[MBTI Detector UI<br/>(GitHub Pages / Vercel / Netlify)]
  ui -->|POST /predict { text }| api[FastAPI API<br/>mbti-detector-api.fly.dev]
  subgraph fly[Fly.io (region: iad)]
    api -->|loads once| model[DistilBERT (Transformers)]
    model --- vol[(Fly Volume<br/>/models)]
    vol <-->|first boot<br/>download| hf[Hugging Face<br/>model repo]
  end
  api -->|CORS OK| ui
  api -->|GET /healthz| health[Health Check]
