from __future__ import annotations

from decimal import Decimal
from typing import Optional

from models import Collection, Customer, Food, Transaction
from rag_service import ByteBitesRAGService


def seed_data() -> tuple[Collection, Customer, Transaction]:
    menu = Collection(
        items=[
            Food(id=1, name="Spicy Burger", price=Decimal("6.99"), category="Entree", popularity=84),
            Food(id=2, name="Large Soda", price=Decimal("2.49"), category="Drinks", popularity=77),
            Food(id=3, name="Chocolate Cake", price=Decimal("4.25"), category="Desserts", popularity=65),
            Food(id=4, name="Fries", price=Decimal("3.49"), category="Sides", popularity=80),
        ]
    )
    customer = Customer(id=1, name="Alice")
    transaction = Transaction(id=1, customer=customer)
    return menu, customer, transaction


def print_menu_items(menu: Collection) -> None:
    print("\nMenu Items")
    print("-" * 45)
    for item in menu.items:
        print(f"{item.id}. {item.name:20} ${item.price:>5}  {item.category:10} pop={item.popularity}")


def print_transaction(transaction: Transaction) -> None:
    print("\nCurrent Transaction")
    print("-" * 45)
    if not transaction.items:
        print("No items yet.")
        return

    for item in transaction.items:
        print(f"- {item.name} (${item.price})")
    print(f"Total: ${transaction.total()}")


def add_item_to_transaction(menu: Collection, transaction: Transaction) -> None:
    print_menu_items(menu)
    raw = input("Enter item ID to add: ").strip()
    if not raw.isdigit():
        print("Invalid ID.")
        return

    item_id = int(raw)
    item = next((food for food in menu.items if food.id == item_id), None)
    if not item:
        print("Item not found.")
        return

    transaction.addItem(item)
    print(f"Added: {item.name}")


def remove_item_from_transaction(transaction: Transaction) -> None:
    raw = input("Enter item ID to remove: ").strip()
    if not raw.isdigit():
        print("Invalid ID.")
        return

    removed = transaction.removeItem(int(raw))
    if removed:
        print("Item removed.")
    else:
        print("Item not found in current transaction.")


def checkout(customer: Customer, transaction: Transaction) -> Transaction:
    if not transaction.items:
        print("Transaction is empty.")
        return transaction

    customer.addPurchase(transaction)
    print(f"Checked out transaction {transaction.id}. Verified user: {customer.isVerified()}")

    next_id = (transaction.id or 1) + 1
    return Transaction(id=next_id, customer=customer)


def ask_ai(rag: ByteBitesRAGService, query: str, menu: Collection, customer: Customer, transaction: Transaction) -> None:
    response = rag.ask(query=query, menu=menu, customer=customer, transaction=transaction)
    print("\nAI Response")
    print("-" * 45)
    print(response.answer)
    print(response.sources)


def ensure_rag_service(rag: Optional[ByteBitesRAGService]) -> Optional[ByteBitesRAGService]:
    if rag is not None:
        return rag

    try:
        return ByteBitesRAGService()
    except RuntimeError as exc:
        print("\nAI is unavailable right now.")
        print(str(exc))
        print("Set GEMINI_API_KEY and try option 6 again.")
        return None


def print_options() -> None:
    print("\nByteBites - Interactive Menu")
    print("-" * 45)
    print("1) Show menu items")
    print("2) Add item to current transaction")
    print("3) Remove item from current transaction")
    print("4) View current transaction")
    print("5) Checkout transaction")
    print("6) Ask Gemini (RAG)")
    print("0) Exit")
    print("Tip: you can also type a natural-language question directly.")


def main() -> None:
    menu, customer, transaction = seed_data()
    rag: Optional[ByteBitesRAGService] = None

    while True:
        print_options()
        command = input("Choose option or ask question: ").strip()

        if command == "0":
            print("Goodbye!")
            break
        if command == "1":
            print_menu_items(menu)
            continue
        if command == "2":
            add_item_to_transaction(menu, transaction)
            continue
        if command == "3":
            remove_item_from_transaction(transaction)
            continue
        if command == "4":
            print_transaction(transaction)
            continue
        if command == "5":
            transaction = checkout(customer, transaction)
            continue
        if command == "6":
            query = input("Ask your question: ").strip()
            if query:
                rag = ensure_rag_service(rag)
                if rag is not None:
                    ask_ai(rag, query, menu, customer, transaction)
            continue

        if command:
            rag = ensure_rag_service(rag)
            if rag is not None:
                ask_ai(rag, command, menu, customer, transaction)


if __name__ == "__main__":
    main()
