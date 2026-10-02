"""Resolve Streamlit Secrets and environment bindings without exposing values."""
import os
import streamlit as st

# Flat deployment names remain supported. Sections are optional Streamlit TOML.
SECTIONS = {
    'OPENAI_API_KEY': ('openai', 'api_key'),
    'LLM_API_KEY': ('openai', 'llm_api_key'),
    'OPENAI_MODEL_NAME': ('openai', 'model'),
    'PINECONE_API_KEY': ('pinecone', 'api_key'),
    'PINECONE_INDEX_NAME': ('pinecone', 'index_name'),
    'PINECONE_NAMESPACE': ('pinecone', 'namespace'),
    'PINECONE_HOST': ('pinecone', 'host'),
    'PINECONE_TEXT_KEY': ('pinecone', 'text_key'),
    'PINECONE_EMBEDDING_MODEL': ('pinecone', 'embedding_model'),
    'PINECONE_EMBEDDING_DIMENSIONS': ('pinecone', 'embedding_dimensions'),
    'CLICKHOUSE_HOST': ('clickhouse', 'host'),
    'CLICKHOUSE_PORT': ('clickhouse', 'port'),
    'CLICKHOUSE_USERNAME': ('clickhouse', 'username'),
    'CLICKHOUSE_PASSWORD': ('clickhouse', 'password'),
    'CLICKHOUSE_DB': ('clickhouse', 'database'),
    'CLICKHOUSE_SECURE': ('clickhouse', 'secure'),
    'CLICKHOUSE_VERIFY': ('clickhouse', 'verify'),
    'CLICKHOUSE_CA_CERT': ('clickhouse', 'ca_cert'),
    'SQLITE_DB_PATH': ('sqlite', 'path'),
}


def get_secret(name, default=''):
    try:
        value = st.secrets.get(name)
        if value is not None:
            return str(value)
        if name in SECTIONS:
            section, field = SECTIONS[name]
            value = st.secrets.get(section, {}).get(field)
            if value is not None:
                return str(value)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    return os.environ.get(name, default)


def enabled(name, default=False):
    return get_secret(name, str(default)).lower() in {'1', 'true', 'yes'}
