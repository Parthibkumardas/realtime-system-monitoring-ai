# Real-time System Monitoring with AI Predictions

An agentic monitoring platform that collects live system metrics, forecasts
resource usage with an ML model, flags anomalies, and lets you ask natural
language questions about incidents through a RAG-powered assistant.

## Architecture

```
frontend (HTML/CSS/JS dashboard)
        |
        v
backend (FastAPI)
  |-- monitoring/   live metric collection (psutil) + rolling time-series store
  |-- models/       anomaly detector (Isolation Forest) + forecaster (LSTM)
  |-- rag/          embeddings + vector store (ChromaDB) + retrieval + LLM answer
  |-- agent/        agentic tool-calling loop that decides which tool(s) to use
  `-- api/          REST routes tying it all together
```

## Features

- **Live metrics** — CPU, memory, disk, network, sampled every few seconds.
- **Forecasting** — an LSTM model predicts CPU/memory 30 minutes ahead.
- **Anomaly detection** — Isolation Forest flags unusual patterns in real time.
- **RAG assistant** — retrieves similar past incidents/runbooks from a vector
  store and answers "why did X happen" questions with cited sources.
- **Agentic layer** — a tool-calling agent decides whether a question needs
  live metrics, a forecast, a RAG lookup, or a combination, then composes the
  final answer.
- **Dashboard** — dark mode, charts (line/pie/histogram/radar), service
  status table, and a chat panel for the assistant.

## Quickstart (local, no Docker)

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env       # fill in your LLM API key
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in a browser (or serve it with any static
server — it calls the backend at `http://localhost:8000`).

## Quickstart (Docker)

```bash
cp .env.example .env             # fill in your LLM API key
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8080
- API docs: http://localhost:8000/docs

## Environment variables

See `.env.example`. At minimum you need an API key for whichever LLM
provider you configure in `backend/config.py` (Anthropic by default).

## Project layout

```
.
├── backend/
│   ├── main.py                  FastAPI app entrypoint
│   ├── config.py                settings loaded from environment
│   ├── requirements.txt
│   ├── monitoring/
│   │   ├── collector.py         psutil-based metric sampling
│   │   └── metrics_store.py     in-memory rolling time-series buffer
│   ├── models/
│   │   ├── anomaly_detector.py  Isolation Forest wrapper
│   │   └── forecaster.py        small PyTorch LSTM for forecasting
│   ├── rag/
│   │   ├── embeddings.py        sentence-transformers embedding wrapper
│   │   ├── vector_store.py      ChromaDB wrapper
│   │   ├── retriever.py         top-k similarity retrieval + formatting
│   │   └── rag_pipeline.py      retrieval + LLM answer generation
│   ├── agent/
│   │   └── agent.py             tool-calling agent loop
│   ├── api/
│   │   └── routes.py            /metrics, /forecast, /anomaly, /rag, /agent
│   └── tests/
│       └── test_api.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/
│   └── incidents.json           sample documents seeded into the vector store
├── .github/workflows/ci-cd.yml  lint + test + docker build/push
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.example
└── .gitignore
```

## Resume-ready summary

> Built a real-time monitoring platform with an LSTM forecasting model,
> Isolation Forest anomaly detection, and a RAG-based assistant (ChromaDB +
> LLM) that answers incident questions grounded in historical logs;
> containerized with Docker and deployed via a GitHub Actions CI/CD pipeline.

## License

MIT
