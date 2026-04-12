import pytest

from llm_client import GeminiClient


def test_gemini_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        GeminiClient()

    assert "GEMINI_API_KEY" in str(exc_info.value)
