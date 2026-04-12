"""Gemini client wrapper for ByteBites RAG queries."""

import os
from typing import Iterable, List, Tuple

from dotenv import load_dotenv

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def resolve_gemini_api_key() -> str:
    """Resolve Gemini API key from environment, loading .env first."""
    load_dotenv()
    return os.getenv("GEMINI_API_KEY", "")


class GeminiClient:
    """Small wrapper around the Google Gen AI SDK for content generation."""

    def __init__(self, model_name: str = GEMINI_MODEL_NAME) -> None:
        api_key = resolve_gemini_api_key()
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it in your shell or .env file to enable LLM features."
            )

        try:
            from google import genai
        except Exception as exc:
            raise RuntimeError(
                "Google Gen AI SDK is not installed. "
                "Install it with: pip install google-genai"
            ) from exc

        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _extract_text(self, response) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", []) or []
        if candidates:
            parts = getattr(candidates[0], "content", None)
            parts = getattr(parts, "parts", []) if parts else []
            chunks = [getattr(part, "text", "") for part in parts if getattr(part, "text", "")]
            if chunks:
                return "\n".join(chunks).strip()

        return ""

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return self._extract_text(response)

    def answer_from_snippets(self, query: str, snippets: Iterable[Tuple[str, str]]) -> str:
        """Answer a user query by grounding on retrieved snippets only."""
        snippets_list: List[Tuple[str, str]] = list(snippets)
        if not snippets_list:
            return "I do not know based on the data I have."

        context_blocks = [f"Source: {source}\n{text}" for source, text in snippets_list]
        context = "\n\n".join(context_blocks)

        prompt = f"""
You are ByteBites assistant. Answer using ONLY the provided context snippets.
If context is insufficient, reply exactly: "I do not know based on the data I have."

Question:
{query}

Context snippets:
{context}

Output requirements:
- Give a concise answer.
- Then list the sources you used in a short line starting with "Sources:".
"""
        return self.generate(prompt).strip()
