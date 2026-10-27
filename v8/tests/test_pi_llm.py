"""PiLLMClient：v8 graph 的默认 LLM 适配器。"""
from __future__ import annotations

from types import SimpleNamespace


def test_pi_llm_client_maps_model_and_system():
    from dag8.pi import PiLLMClient

    calls = []

    class Runner:
        def run(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return "ok"

    client = PiLLMClient(provider="laptop", agent=Runner(), timeout=12)
    model = SimpleNamespace(key="sensenova", model="glm-5.2")
    out = client.complete("问题", model_config=model, system_prompt="系统")
    assert out == "ok"
    prompt, kwargs = calls[0]
    assert "系统" in prompt and "问题" in prompt
    assert kwargs["provider"] == "sensenova"
    assert kwargs["model"] == "glm-5.2"
    assert kwargs["timeout"] == 12


def test_pi_llm_client_embed_is_deterministic():
    from dag8.pi import PiLLMClient

    client = PiLLMClient(agent=object())
    a = client.embed("收入")
    b = client.embed("收入")
    assert a == b and len(a) == 32
