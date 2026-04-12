from decimal import Decimal

from models import Collection, Customer, Food, Transaction
from rag_retriever import RuntimeRetriever, as_snippets, format_sources
from rag_service import ByteBitesRAGService


class FakeLLMClient:
    def answer_from_snippets(self, query, snippets):
        if not snippets:
            return "I do not know based on the data I have."
        first_source, _ = snippets[0]
        return f"Answer for '{query}' from {first_source}"


def test_runtime_retriever_finds_relevant_food_chunk():
    menu = Collection(
        items=[
            Food(id=1, name="Spicy Burger", price=Decimal("6.99"), category="Entree", popularity=84),
            Food(id=2, name="Large Soda", price=Decimal("2.49"), category="Drinks", popularity=77),
        ]
    )
    customer = Customer(id=1, name="Alice")
    tx = Transaction(id=1, customer=customer)

    retriever = RuntimeRetriever()
    results = retriever.retrieve("What is the price of Spicy Burger?", menu, customer, tx, k=3)

    assert results
    sources = [result.chunk.source for result in results]
    assert any("Spicy Burger" in source for source in sources)


def test_rag_service_returns_answer_and_citations():
    menu = Collection(items=[Food(id=1, name="Fries", price=Decimal("3.49"), category="Sides", popularity=80)])
    customer = Customer(id=1, name="Alice")
    tx = Transaction(id=1, customer=customer)

    rag = ByteBitesRAGService(llm_client=FakeLLMClient(), retriever=RuntimeRetriever())
    response = rag.ask("Tell me about fries", menu, customer, tx)

    assert "Answer for" in response.answer
    assert response.sources.startswith("Sources:")


def test_snippet_helpers_format_sources():
    menu = Collection(items=[Food(id=5, name="Cake", price=Decimal("4.25"), category="Desserts", popularity=65)])
    retriever = RuntimeRetriever()
    results = retriever.retrieve("cake dessert", menu, None, None, k=2)

    snippets = as_snippets(results)
    assert snippets

    src_line = format_sources(results)
    assert src_line.startswith("Sources:")
    assert "food:" in src_line
