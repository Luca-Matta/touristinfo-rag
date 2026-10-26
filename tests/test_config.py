import pytest

from rag.config import Settings
from rag.errors import ConfigError


def test_auto_providers_follow_the_keys_present():
    assert Settings.from_env({}).llm_provider is None
    assert Settings.from_env({}).embedding_provider == "local"
    s = Settings.from_env({"OPENAI_API_KEY": "sk-x"})
    assert (s.llm_provider, s.embedding_provider) == ("openai", "openai")
    s = Settings.from_env({"ANTHROPIC_API_KEY": "sk-ant-x"})
    assert (s.llm_provider, s.embedding_provider) == ("anthropic", "local")


def test_explicit_choice_and_validation():
    s = Settings.from_env({"OPENAI_API_KEY": "k", "EMBEDDING_PROVIDER": "local", "TOP_K": "3"})
    assert s.embedding_provider == "local" and s.top_k == 3
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_PROVIDER": "mistral"})
