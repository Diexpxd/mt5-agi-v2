"""Cliente LLM con bypass: Gemini si hay clave real; ``MockLLM`` determinista en caso contrario."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from config.settings import Settings, is_mock_key

log = logging.getLogger(__name__)


class LLMClient:
    """Interfaz mínima."""

    is_mock = True
    name = "base"

    def generate(self, prompt: str, system: str | None = None) -> str:  # pragma: no cover - interfaz
        raise NotImplementedError

    def generate_json(self, prompt: str, system: str | None = None) -> Optional[Dict[str, Any]]:
        try:
            return extract_json(self.generate(prompt, system))
        except Exception as exc:
            log.warning("LLM generate_json falló: %s", exc)
            return None


class MockLLM(LLMClient):
    """Sin red ni claves: devuelve un texto fijo. Los agentes lo detectan (``is_mock``) y usan sus fallbacks."""

    is_mock = True
    name = "mock"

    def generate(self, prompt: str, system: str | None = None) -> str:
        return "[MOCK-LLM] respuesta simulada"


class GeminiClient(LLMClient):
    """Gemini vía ``google-genai``."""

    is_mock = False
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, system: str | None = None) -> str:
        cfg: Dict[str, Any] = {"temperature": 0.2}
        if system:
            cfg["system_instruction"] = system
        res = self._client.models.generate_content(model=self._model, contents=prompt, config=cfg)
        return res.text or ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """Extrae el primer objeto JSON de ``text`` (tolera ```json ... ``` y texto alrededor)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    m = _JSON_RE.search(cleaned)
    if not m:
        raise ValueError("sin JSON en la respuesta")
    return json.loads(m.group(0))


def make_llm(settings: Settings) -> LLMClient:
    """Devuelve ``GeminiClient`` si hay clave real y el SDK carga; si no, ``MockLLM``."""
    if is_mock_key(settings.gemini_api_key):
        log.info("GEMINI_API_KEY ausente/MOCK: LLM en modo bypass")
        return MockLLM()
    try:
        return GeminiClient(settings.gemini_api_key, settings.gemini_model)
    except Exception as exc:
        log.warning("No se pudo iniciar Gemini (%s): bypass activado", exc)
        return MockLLM()
