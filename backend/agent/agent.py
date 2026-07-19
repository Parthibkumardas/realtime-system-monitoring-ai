"""
A small agentic loop: the LLM is given a set of tools (get_live_metrics,
get_forecast, get_anomaly_status, search_incidents) and decides which ones
to call, in what order, before composing a final answer. This is what
turns the assistant from "one RAG lookup" into something that can reason
about which information it actually needs.
"""
import json

from config import settings

TOOLS = [
    {
        "name": "get_live_metrics",
        "description": "Get the latest CPU, memory, disk, and network metrics.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_forecast",
        "description": "Get the forecasted CPU usage for the next several minutes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_anomaly_status",
        "description": "Check whether the most recent metric sample is anomalous.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_incidents",
        "description": "Search historical incidents and runbooks relevant to a question.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

AGENT_SYSTEM_PROMPT = """You are an ops assistant for a real-time system monitoring \
platform. Use the available tools to gather whatever live data or historical \
context you need before answering. Prefer calling tools over guessing. Keep the \
final answer concise and actionable."""


class MonitoringAgent:
    def __init__(self, metrics_store, anomaly_detector, forecaster, retriever):
        self.metrics_store = metrics_store
        self.anomaly_detector = anomaly_detector
        self.forecaster = forecaster
        self.retriever = retriever
        self._client = self._build_client()

    def _build_client(self):
        import anthropic

        return anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # ---- tool implementations -------------------------------------------------

    def _tool_get_live_metrics(self, _args: dict) -> dict:
        latest = self.metrics_store.latest()
        return latest or {"error": "no metrics collected yet"}

    def _tool_get_forecast(self, _args: dict) -> dict:
        series = self.metrics_store.as_series("cpu_percent")
        prediction = self.forecaster.predict_next(series)
        return {"cpu_forecast_next_steps": prediction}

    def _tool_get_anomaly_status(self, _args: dict) -> dict:
        samples = self.metrics_store.recent()
        return self.anomaly_detector.score_latest(samples)

    def _tool_search_incidents(self, args: dict) -> dict:
        query = args.get("query", "")
        result = self.retriever.retrieve(query)
        return result

    def _dispatch(self, tool_name: str, tool_input: dict) -> dict:
        handlers = {
            "get_live_metrics": self._tool_get_live_metrics,
            "get_forecast": self._tool_get_forecast,
            "get_anomaly_status": self._tool_get_anomaly_status,
            "search_incidents": self._tool_search_incidents,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"unknown tool {tool_name}"}
        return handler(tool_input)

    # ---- agent loop -------------------------------------------------------

    def run(self, user_question: str, max_turns: int = 4) -> dict:
        messages = [{"role": "user", "content": user_question}]
        tool_calls_made = []

        for _ in range(max_turns):
            response = self._client.messages.create(
                model=settings.llm_model,
                max_tokens=800,
                system=AGENT_SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                return {"answer": final_text, "tool_calls": tool_calls_made}

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = self._dispatch(block.name, block.input)
                tool_calls_made.append({"tool": block.name, "input": block.input})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        return {
            "answer": "Reached the maximum number of reasoning steps without a final answer.",
            "tool_calls": tool_calls_made,
        }
