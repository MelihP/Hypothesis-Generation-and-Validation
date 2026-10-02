import os
import streamlit as st


def get_secret(name, default=""):
    try:
        value = st.secrets.get(name)
        if value is not None:
            return str(value)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    return os.environ.get(name, default)


def enabled(name, default=False):
    return get_secret(name, str(default)).lower() in {"1", "true", "yes"}
