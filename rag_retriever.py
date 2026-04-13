from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional

from models import Collection, Customer, Transaction


@dataclass
class Chunk:
    source: str
    text: str
    metadata: Dict[str, str]


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: int


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

_SYNONYMS: Dict[str, set[str]] = {
    "order": {"orders", "ordered", "purchase", "purchases", "transaction", "transactions", "items"},
    "today": {"current", "now", "latest", "recent", "transaction"},
    "bought": {"purchase", "purchases", "ordered", "order", "transaction"},
    "menu": {"items", "food", "foods"},
    "price": {"cost", "total"},
}

_ORDER_INTENT_TOKENS = {
    "order",
    "orders",
    "ordered",
    "purchase",
    "purchases",
    "bought",
    "transaction",
    "transactions",
    "history",
}


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}


def _expand_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(_SYNONYMS.get(token, set()))
    return expanded


def _intent_flags(query_tokens: set[str]) -> Dict[str, bool]:
    return {
        "order": bool(query_tokens.intersection(_ORDER_INTENT_TOKENS)),
        "today": "today" in query_tokens,
    }


class RuntimeRetriever:
    """Retrieves relevant chunks from in-memory ByteBites runtime objects."""

    def build_corpus(
        self,
        menu: Collection,
        customer: Optional[Customer],
        transaction: Optional[Transaction],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []

        for food in menu.items:
            chunks.append(
                Chunk(
                    source=f"food:{food.id}:{food.name}",
                    text=(
                        f"Food item {food.name} has id {food.id}, category {food.category}, "
                        f"price ${food.price}, popularity {food.popularity}."
                    ),
                    metadata={"type": "food", "id": str(food.id), "name": food.name},
                )
            )

        if customer:
            chunks.append(
                Chunk(
                    source=f"customer:{customer.id}:{customer.name}",
                    text=(
                        f"Customer {customer.name} has id {customer.id} and "
                        f"{len(customer.purchaseHistory)} purchases."
                    ),
                    metadata={"type": "customer", "id": str(customer.id), "name": customer.name},
                )
            )

            for purchase in customer.purchaseHistory:
                item_names = ", ".join([item.name for item in purchase.items]) or "no items"
                chunks.append(
                    Chunk(
                        source=f"purchase:{purchase.id or 'unknown'}",
                        text=(
                            f"Purchase {purchase.id or 'unknown'} includes items: {item_names}. "
                            f"Total ${purchase.total()}."
                        ),
                        metadata={"type": "purchase", "id": str(purchase.id or "unknown")},
                    )
                )

        if transaction:
            current_items = ", ".join([item.name for item in transaction.items]) or "no items"
            chunks.append(
                Chunk(
                    source=f"transaction:{transaction.id or 'current'}",
                    text=f"Current transaction has items: {current_items}. Total ${transaction.total()}.",
                    metadata={"type": "transaction", "id": str(transaction.id or "current")},
                )
            )

        return chunks

    def retrieve(
        self,
        query: str,
        menu: Collection,
        customer: Optional[Customer],
        transaction: Optional[Transaction],
        k: int = 5,
    ) -> List[RetrievalResult]:
        corpus = self.build_corpus(menu, customer, transaction)
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        expanded_query_tokens = _expand_tokens(query_tokens)
        flags = _intent_flags(query_tokens)

        scored: List[RetrievalResult] = []
        for chunk in corpus:
            chunk_tokens = _tokenize(chunk.text)

            exact_matches = len(query_tokens.intersection(chunk_tokens))
            expanded_matches = len(expanded_query_tokens.intersection(chunk_tokens)) - exact_matches
            score = (exact_matches * 2) + max(0, expanded_matches)

            chunk_type = chunk.metadata.get("type", "")

            if flags["order"] and chunk_type in {"purchase", "transaction"}:
                score += 2
            if flags["today"] and chunk_type == "transaction":
                score += 3
            if flags["today"] and chunk_type == "purchase":
                score += 1

            if score > 0:
                scored.append(RetrievalResult(chunk=chunk, score=score))

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:k]


def as_snippets(results: Iterable[RetrievalResult]) -> List[tuple[str, str]]:
    return [(result.chunk.source, result.chunk.text) for result in results]


def format_sources(results: Iterable[RetrievalResult]) -> str:
    labels = [result.chunk.source for result in results]
    if not labels:
        return "Sources: none"
    return f"Sources: {', '.join(labels)}"
