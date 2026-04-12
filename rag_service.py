from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_client import GeminiClient
from models import Collection, Customer, Transaction
from rag_retriever import RuntimeRetriever, as_snippets, format_sources


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
        sources = format_sources(results)
        return RAGResponse(answer=answer, sources=sources)
