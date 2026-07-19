import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.agent import MonitoringAgent
from api.routes import router
from config import settings
from models.anomaly_detector import AnomalyDetector
from models.forecaster import Forecaster
from monitoring.collector import MetricCollector
from monitoring.metrics_store import MetricsStore
from rag.embeddings import EmbeddingModel
from rag.rag_pipeline import RAGPipeline
from rag.retriever import Retriever
from rag.vector_store import VectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def seed_vector_store(vector_store: VectorStore) -> None:
    """Load sample incidents/runbooks into Chroma if it's empty."""
    if not vector_store.is_empty():
        return

    incidents_path = DATA_DIR / "incidents.json"
    if not incidents_path.exists():
        return

    with open(incidents_path) as f:
        raw_docs = json.load(f)

    docs = [
        {
            "id": d["id"],
            "text": d["text"],
            "metadata": {"source": d["id"], "service": d.get("service", "unknown")},
        }
        for d in raw_docs
    ]
    vector_store.add_documents(docs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- build components -------------------------------------------------
    metrics_store = MetricsStore(max_length=settings.metric_history_length)
    collector = MetricCollector(metrics_store, interval_seconds=settings.metric_sample_interval_seconds)
    anomaly_detector = AnomalyDetector(contamination=settings.anomaly_contamination)
    forecaster = Forecaster()

    embedder = EmbeddingModel(settings.embedding_model)
    vector_store = VectorStore(settings.chroma_persist_dir, embedder)
    seed_vector_store(vector_store)
    retriever = Retriever(vector_store)
    rag_pipeline = RAGPipeline(retriever)
    agent = MonitoringAgent(metrics_store, anomaly_detector, forecaster, retriever)

    # --- attach to app state so routes can reach them ----------------------
    app.state.metrics_store = metrics_store
    app.state.anomaly_detector = anomaly_detector
    app.state.forecaster = forecaster
    app.state.rag_pipeline = rag_pipeline
    app.state.agent = agent

    collector.start()
    yield
    collector.stop()


app = FastAPI(title="Real-time System Monitoring with AI Predictions", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
