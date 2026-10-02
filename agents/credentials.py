"""Resolve OpenAI bindings; refreshed Streamlit Secrets take precedence."""
import os
from agents.config import get_secret


def configure_openai_credentials() -> None:
    key = get_secret("OPENAI_API_KEY") or get_secret("LLM_API_KEY")
    if key:
        os.environ["OPENAI_API_KEY"] = key
