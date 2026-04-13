# ByteBites Gemini RAG

## Project Source

I am using the Tinker: ByteBits from Week 3 (Module 2). The original project was very simple and could only create customer, food, transaction and retrive them and no interactive interface. 

## Title and Summary

ByteBites Gemini RAG is a command-line restaurant assistant that combines app state with Retrieval-Augmented Generation (RAG) to answer user questions about menu items, pricing, and order context.

This matters because the assistant is grounded in real runtime data (menu, customer, transactions) instead of relying only on model memory, which improves reliability and traceability through source citations.

## Architecture Overview

The system is shown in [assets/system_diagram.md](assets/system_diagram.md).

- `main.py` handles user input (menu actions or natural-language questions).
- `ByteBitesRAGService` acts as the orchestration layer (agent).
- `RuntimeRetriever` retrieves top relevant chunks from in-memory runtime objects.
- `GeminiClient` generates grounded answers from retrieved snippets.
- Output includes both the answer and `Sources: ...` citations.
- Human-in-the-loop check happens when users review responses; automated tests validate retriever and orchestration behavior.

## Setup Instructions

1. Open the project folder.
2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Set your Gemini API key:

	```bash
	export GEMINI_API_KEY="your-api-key"
	```

4. (Optional) Set model name:

	```bash
	export GEMINI_MODEL="gemini-2.5-flash"
	```

5. Run the app:

	```bash
	python main.py
	```

6. Use either numbered menu options or type natural-language questions directly.

## Sample Interactions

Example 1: Price lookup

- Input: `What is the price of Spicy Burger?`
- Output (example): `Spicy Burger is $6.99.`
- Sources (example): `Sources: food:1:Spicy Burger`

Example 2: Recommendation fallback

- Input: `what should i get with the spicy burger?`
- Output (example): `Try Spicy Burger with Fries and Large Soda. These are strong companion picks based on popularity.`
- Sources (example): `Sources: food:1:Spicy Burger, food:2:Fries, food:3:Large Soda`

Example 3: Current order context

- Input: `what did I order today?`
- Output (example): `Your current transaction includes Spicy Burger and Large Soda.`
- Sources (example): `Sources: transaction:10`

Note: LLM phrasing can vary between runs, but retrieval sources and grounding behavior should remain consistent.

## Design Decisions

- Runtime in-memory retrieval keeps architecture simple and fast for this project scope.
- Lexical token scoring was chosen over embeddings to reduce complexity and dependency overhead.
- A small recommendation fallback handles low-confidence model responses (`I do not know ...`) for better UX.
- Citations were included to make outputs inspectable and easier to trust.

Trade-offs:

- Lexical retrieval is lightweight but less semantically rich than embedding-based retrieval.
- In-memory corpus is easy to maintain but not ideal for very large datasets or multi-user persistence.
- Rule-based fallback improves reliability for specific intents but is less flexible than fully learned policies.

## Testing Summary

What worked:

- Targeted RAG tests passed using the project conda interpreter:

  ```bash
  /Users/djamellhermitus/opt/anaconda3/envs/codepathai/bin/python -m pytest -q test_rag.py
  ```

- Result: `5 passed in 0.01s`
- Verified behaviors: relevant retrieval, citation formatting, orchestration response shape, order-intent prioritization, and recommendation fallback.

What did not work:

- You can not use the AI to change current data, only retrival is possible.

What I learned:

- Interpreter-specific test commands are more reliable than assuming global PATH tools.
- Grounded outputs with source labels are valuable for debugging and user trust.
- Small deterministic fallbacks can improve perceived quality without major architectural complexity.

## Reflection

This was really fun in adding new feature while trying to keep it simple and understandable. The retrival function had some challenges that limited the amount of data passed to the AI, thus limiting the usefulness in some cases.
