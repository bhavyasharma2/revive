from __future__ import annotations

from types import SimpleNamespace

import pytest

from revive.llm import (
    EmptyCompletion,
    MissingAPIKey,
    ModelNotAvailable,
    TokenFactoryClient,
    pick_model,
)

MODELS = [
    "deepseek-ai/DeepSeek-R1-0528",
    "meta-llama/Llama-3.3-70B-Instruct",
    "nvidia/Nemotron-3-Nano-Omni",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
]


def test_pick_model_by_role():
    assert pick_model(MODELS, "deep") == "nvidia/Nemotron-3-Ultra-550b-a55b"
    assert pick_model(MODELS, "balanced") == "nvidia/nemotron-3-super-120b-a12b"


# The exact list Token Factory returned on 2026-10-23.
LIVE_NVIDIA_MODELS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
    "nvidia/Nemotron-3.5-Super-VL",
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/nemotron-3-super-120b-a12b",
]


def test_live_model_list_resolves_to_text_models():
    assert pick_model(LIVE_NVIDIA_MODELS, "deep") == "nvidia/Nemotron-3-Ultra-550b-a55b"
    assert pick_model(LIVE_NVIDIA_MODELS, "balanced") == "nvidia/nemotron-3-super-120b-a12b"
    assert pick_model(LIVE_NVIDIA_MODELS, "fast") == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def test_balanced_falls_back_to_vl_if_it_is_the_only_super():
    only_vl = [m for m in LIVE_NVIDIA_MODELS if "120b" not in m]
    assert pick_model(only_vl, "balanced") == "nvidia/Nemotron-3.5-Super-VL"


def test_fast_role_prefers_plain_nano_over_omni():
    assert pick_model(MODELS, "fast") == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def test_fast_role_falls_back_to_omni_if_it_is_the_only_nano():
    only_omni = [m for m in MODELS if "30B" not in m]
    assert pick_model(only_omni, "fast") == "nvidia/Nemotron-3-Nano-Omni"


def test_override_must_exist_and_is_case_insensitive():
    assert pick_model(MODELS, "deep", "NVIDIA/nemotron-3-super-120B-A12B") == "nvidia/nemotron-3-super-120b-a12b"
    with pytest.raises(ModelNotAvailable):
        pick_model(MODELS, "deep", "nvidia/does-not-exist")


def test_missing_role_model_gives_helpful_error():
    with pytest.raises(ModelNotAvailable) as err:
        pick_model(["meta-llama/Llama-3.3-70B-Instruct"], "deep")
    assert "ultra" in str(err.value)


def test_unknown_role():
    with pytest.raises(ModelNotAvailable):
        pick_model(MODELS, "genius")


class FakeOpenAI:
    """Mimics the parts of the OpenAI SDK that TokenFactoryClient uses."""

    def __init__(self, models=MODELS, content="<think>hmm</think>pong", finish_reason="stop", choices=True):
        self.list_calls = 0
        self.requests = []
        self._models = models
        self._content = content
        self._finish = finish_reason
        self._choices = choices
        self.models = SimpleNamespace(list=self._list)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _list(self):
        self.list_calls += 1
        return [SimpleNamespace(id=m) for m in self._models]

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._choices:
            return SimpleNamespace(choices=[])
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self._finish)])


def test_models_are_listed_once_and_cached():
    fake = FakeOpenAI()
    client = TokenFactoryClient(client=fake)
    client.available_models()
    client.model_for("deep")
    client.model_for("fast")
    assert fake.list_calls == 1
    client.available_models(refresh=True)
    assert fake.list_calls == 2


def test_chat_resolves_role_strips_reasoning_and_sends_expected_request():
    fake = FakeOpenAI()
    client = TokenFactoryClient(client=fake)
    reply = client.chat([{"role": "user", "content": "ping"}], role="deep", max_tokens=100, temperature=0.0)
    assert reply == "pong"
    request = fake.requests[0]
    assert request["model"] == "nvidia/Nemotron-3-Ultra-550b-a55b"
    assert request["max_tokens"] == 100 and request["temperature"] == 0.0
    assert "extra_body" not in request


def test_usage_is_accumulated_per_model():
    fake = FakeOpenAI()
    original = fake._create

    def with_usage(**kwargs):
        response = original(**kwargs)
        response.usage = SimpleNamespace(prompt_tokens=100, completion_tokens=40)
        return response

    fake.chat.completions.create = with_usage
    client = TokenFactoryClient(client=fake)
    client.chat([{"role": "user", "content": "x"}], role="deep")
    client.chat([{"role": "user", "content": "x"}], role="deep")
    client.chat([{"role": "user", "content": "x"}], role="fast")
    assert client.usage["nvidia/Nemotron-3-Ultra-550b-a55b"] == {"calls": 2, "prompt_tokens": 200, "completion_tokens": 80}
    assert client.usage["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"]["calls"] == 1
    client.reset_usage()
    assert client.usage == {}


def test_usage_tolerates_missing_usage_field():
    client = TokenFactoryClient(client=FakeOpenAI())
    client.chat([{"role": "user", "content": "x"}], role="deep")
    assert client.usage["nvidia/Nemotron-3-Ultra-550b-a55b"] == {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0}


def test_chat_passes_explicit_model_and_extra_body():
    fake = FakeOpenAI()
    client = TokenFactoryClient(client=fake)
    client.chat([{"role": "user", "content": "x"}], model="custom/model", extra_body={"a": 1})
    assert fake.requests[0]["model"] == "custom/model"
    assert fake.requests[0]["extra_body"] == {"a": 1}
    assert fake.list_calls == 0  # explicit model needs no lookup


def test_env_override_pins_a_role(monkeypatch):
    monkeypatch.setenv("REVIVE_MODEL_DEEP", "nvidia/nemotron-3-super-120b-a12b")
    client = TokenFactoryClient(client=FakeOpenAI())
    assert client.model_for("deep") == "nvidia/nemotron-3-super-120b-a12b"


def test_empty_completion_raises_with_length_hint():
    client = TokenFactoryClient(client=FakeOpenAI(content="<think>only thinking", finish_reason="length"))
    with pytest.raises(EmptyCompletion) as err:
        client.chat([{"role": "user", "content": "x"}], role="deep")
    assert "max_tokens" in str(err.value)


def test_no_choices_raises():
    client = TokenFactoryClient(client=FakeOpenAI(choices=False))
    with pytest.raises(EmptyCompletion):
        client.chat([{"role": "user", "content": "x"}], role="deep")


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey):
        TokenFactoryClient()


def test_real_openai_client_is_pointed_at_token_factory(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    client = TokenFactoryClient()
    assert str(client._client.base_url).startswith("https://api.tokenfactory.nebius.com/v1")
