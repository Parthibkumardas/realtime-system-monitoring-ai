"""
Combines retrieval with an LLM call to produce a grounded answer.
Supports Anthropic and OpenAI; pick the provider via LLM_PROVIDER in .env.
"""
from config import settings
from rag.retriever import Retriever

SYSTEM_PROMPT = """You are a system-monitoring assistant. Answer questions about \
anomalies and incidents using ONLY the retrieved context provided. If the \
context doesn't contain a clear answer, say what you don't know rather than \
guessing. Be concise: 3-5 sentences plus a short recommendation if relevant."""


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self._client = self._build_client()

    def _build_client(self):
        if settings.llm_provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=settings.anthropic_api_key)
        elif settings.llm_provider == "openai":
            import openai

            return openai.OpenAI(api_key=settings.openai_api_key)
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

    def _call_llm(self, user_prompt: str) -> str:
        if settings.llm_provider == "anthropic":
            response = self._client.messages.create(
                model=settings.llm_model,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(block.text for block in response.content if block.type == "text")

        response = self._client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def answer(self, question: str, live_context: str | None = None) -> dict:
        retrieval = self.retriever.retrieve(question)

        prompt_parts = [f"Question: {question}"]
        if live_context:
            prompt_parts.append(f"\nCurrent system state:\n{live_context}")
        if retrieval["context"]:
            prompt_parts.append(f"\nRetrieved context:\n{retrieval['context']}")
        else:
            prompt_parts.append("\nNo relevant historical context was retrieved.")

        answer_text = self._call_llm("\n".join(prompt_parts))

        return {
            "answer": answer_text,
            "citations": retrieval["citations"],
            "hit_count": retrieval["hit_count"],
        }
