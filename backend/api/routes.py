from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class RagQuery(BaseModel):
    question: str


class AgentQuery(BaseModel):
    question: str


@router.get("/metrics/latest")
def get_latest_metrics(request: Request):
    store = request.app.state.metrics_store
    latest = store.latest()
    if latest is None:
        raise HTTPException(status_code=503, detail="no metrics collected yet")
    return latest


@router.get("/metrics/history")
def get_metrics_history(request: Request, n: int = 60):
    store = request.app.state.metrics_store
    return {"samples": store.recent(n)}


@router.get("/forecast/cpu")
def get_cpu_forecast(request: Request):
    store = request.app.state.metrics_store
    forecaster = request.app.state.forecaster
    series = store.as_series("cpu_percent")

    if not forecaster.trained:
        trained = forecaster.train(series)
        if not trained:
            raise HTTPException(status_code=503, detail="not enough history to train forecaster yet")

    prediction = forecaster.predict_next(series)
    return {"history": series[-24:], "forecast": prediction}


@router.get("/anomaly/status")
def get_anomaly_status(request: Request):
    store = request.app.state.metrics_store
    detector = request.app.state.anomaly_detector
    samples = store.recent()
    return detector.score_latest(samples)


@router.post("/rag/query")
def rag_query(payload: RagQuery, request: Request):
    pipeline = request.app.state.rag_pipeline
    store = request.app.state.metrics_store

    latest = store.latest()
    live_context = None
    if latest:
        live_context = (
            f"cpu={latest['cpu_percent']}%, memory={latest['memory_percent']}%, "
            f"disk={latest['disk_percent']}%"
        )

    return pipeline.answer(payload.question, live_context=live_context)


@router.post("/agent/query")
def agent_query(payload: AgentQuery, request: Request):
    agent = request.app.state.agent
    return agent.run(payload.question)


@router.get("/health")
def health():
    return {"status": "ok"}
