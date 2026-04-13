from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from llm_client import GeminiClient
from models import Collection, Customer, Transaction
from rag_retriever import RuntimeRetriever, as_snippets, format_sources


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_UNKNOWN_ANSWER = "I do not know based on the data I have."
_RECOMMENDATION_TOKENS = {"recommend", "suggest", "pair", "pairing", "with", "combo", "get"}


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}


@dataclass
class RAGResponse:
    answer: str
    sources: str


class ByteBitesRAGService:
    def __init__(self, llm_client: Optional[GeminiClient] = None, retriever: Optional[RuntimeRetriever] = None) -> None:
        self.llm_client = llm_client or GeminiClient()
        self.retriever = retriever or RuntimeRetriever()

    def ask(
        self,
        query: str,
        menu: Collection,
        customer: Optional[Customer],
        transaction: Optional[Transaction],
        k: int = 5,
    ) -> RAGResponse:
        results = self.retriever.retrieve(query, menu, customer, transaction, k=k)
        snippets = as_snippets(results)
        answer = self.llm_client.answer_from_snippets(query, snippets)
        if answer.strip() == _UNKNOWN_ANSWER and snippets:
            fallback = self._recommendation_fallback(query, menu)
            if fallback:
                answer = fallback
        sources = format_sources(results)
        return RAGResponse(answer=answer, sources=sources)

    def _recommendation_fallback(self, query: str, menu: Collection) -> Optional[str]:
        query_lower = (query or "").lower()
        query_tokens = _tokenize(query_lower)
        if not query_tokens.intersection(_RECOMMENDATION_TOKENS):
            return None

        if not menu.items:
            return None

        mentioned = [food for food in menu.items if food.name and food.name.lower() in query_lower]
        primary = mentioned[0] if mentioned else None

        candidates = [food for food in menu.items if primary is None or food.id != primary.id]
        if primary and primary.category:
            different_category = [food for food in candidates if food.category != primary.category]
            if different_category:
                candidates = different_category

        ranked = sorted(candidates, key=lambda food: food.popularity, reverse=True)
        picks = ranked[:2]
        if not picks:
            return None

        pick_names = " and ".join(food.name for food in picks)
        if primary:
            return (
                f"Try {primary.name} with {pick_names}. "
                "These are strong companion picks based on popularity."
            )

        return f"Good picks are {pick_names}, based on current menu popularity."
