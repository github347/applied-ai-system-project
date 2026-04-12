import pytest

from llm_client import GeminiClient, resolve_gemini_api_key


def test_gemini_client_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        GeminiClient()

    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_resolve_gemini_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert resolve_gemini_api_key() == "from-env"


def test_resolve_gemini_api_key_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resolve_gemini_api_key() == "from-dotenv"
