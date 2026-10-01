"""Resolve supported OpenAI bindings without logging or persisting secrets."""
import os

import streamlit as st


def configure_openai_credentials() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    if os.environ.get("LLM_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]
        return
    try:
        for name in ("OPENAI_API_KEY", "LLM_API_KEY"):
            if st.secrets.get(name):
                os.environ["OPENAI_API_KEY"] = st.secrets[name]
                return
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
