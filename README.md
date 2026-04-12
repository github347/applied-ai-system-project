## Weekly TF Task 3

**The core concept students needed to understand** is how to interpret client requirements and implement them properly with the help of AI. **Students are most likely to struggle with** keeping the AI on track and avoid implementing features that are not requested by the client. They might also struggle with the models created and how they are supposed to interact with each other like one to many, vice versa or  many to many. **The AI was helpful** in helping plan out the models based on the specs. **However**, the generated code as I left it in mine is way too complex and could be simplified and more user friendly. **One way I would guide a student without giving the answer** is to encourage them to keep refining their answers and when the AI generates something you don’t understand, ask it to explain what is going on as much as needed.

## ByteBites Gemini RAG

This project now includes a runtime Retrieval-Augmented Generation (RAG) flow powered by Gemini.

- Runtime data source: menu items, customer state, and current/past transactions
- Retrieval: lexical top-k chunk retrieval over in-memory app objects
- Generation: Gemini (`gemini-2.5-flash` by default) answers grounded on retrieved snippets
- Output: answer + source citations

### Setup

1. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

2. Export your Gemini key:

	```bash
	export GEMINI_API_KEY="your-api-key"
	```

Optional:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
```

### Run interactive app

```bash
python main.py
```

You can:
- Choose numeric options from the menu
- Type natural-language questions directly (RAG query path)

### Run tests

```bash
pytest -q
```
