from __future__ import annotations

from agents.model import DeterministicModel, GeminiModel, get_model


def test_get_model_defaults_to_deterministic(monkeypatch):
    monkeypatch.delenv("AGENT_OS_LLM_ENABLED", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    model = get_model()

    assert isinstance(model, DeterministicModel)
    assert model.name == "deterministic"


def test_get_model_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("AGENT_OS_LLM_ENABLED", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    model = get_model()

    assert isinstance(model, DeterministicModel)


def test_get_model_uses_gemini_when_enabled(monkeypatch):
    monkeypatch.setenv("AGENT_OS_LLM_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    model = get_model()

    assert isinstance(model, GeminiModel)
    assert model.name == "gemini"
