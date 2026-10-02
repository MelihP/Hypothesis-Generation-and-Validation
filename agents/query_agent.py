"""
Sorgu Planlama Ajanı (Query Planning Agent)
Doğal dil sorgularını yapılandırılmış JSON formatına dönüştürür ve deterministik SQL derleyicisini kullanarak çalıştırır.
"""

import os
import json
import logging
import re
from typing import Any, Optional
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agents.database import adapt_database, SQLiteBackend
from agents.data_schema import validate_query_schema
from agents.sql_safety import validate_plan, DEFAULT_ROWS
from agents.credentials import configure_openai_credentials
from pathlib import Path
import copy
import streamlit as st

from agents.sql_compiler import (
    compile_json_to_sql,
    quote_ident,
    _lit,
    _contains_lit,
    _format_array_lit,
    _compile_join,
    _FILTER_OP_TO_SQL,
    _AGG_OPS,
)

logger = logging.getLogger(__name__)

CLICKHOUSE_CORE_TABLES = ["tweet_predictions", "tweets", "users", "user_factors"]

# --- 1. API ANAHTARLARI (Streamlit Secrets & Ortam Değişkenleri ile Uyumlu) ---
try:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass


from agents.prompts.query_prompts import QUERY_GENERATOR_SYSTEM_PROMPT

# Geriye dönük uyumluluk için alias
_QUERY_GENERATOR_SYSTEM_PROMPT = QUERY_GENERATOR_SYSTEM_PROMPT



# --- 3. SORGU PLANLAMA AJANI SINIFI (Lazy-Loaded LLM & DB) ---

import time

try:
    from logger import log_query
except ImportError:
    try:
        from ..logger import log_query
    except Exception:
        def log_query(*args, **kwargs):
            pass


class QueryAgent:
    """
    Doğal dil sorularını yapılandırılmış JSON formatına çeviren ve deterministik SQL üreten sorgu ajanı.
    """
    def __init__(
        self,
        db_uri: str = "sqlite:///insight_generation_bot.db",
        model_name: str = "gpt-4o",
        dialect: str = "sqlite",
        api_key: Optional[str] = None,
        llm: Optional[Any] = None,
        db: Optional[Any] = None,
        schema: Optional[str] = None
    ):
        self.db_uri = db_uri
        self.model_name = model_name
        self.dialect = dialect
        self.api_key = api_key
        self._db = db
        self._llm = llm
        self._schema = schema

    @property
    def db(self) -> SQLDatabase:
        if self._db is None:
            if self.dialect == "clickhouse" or "clickhouse" in self.db_uri:
                from agent import get_database_connection
                self._db, _, self.dialect = get_database_connection(self.db_uri)
            else:
                from sqlalchemy.engine import make_url
                name = make_url(self.db_uri).database
                if not name:
                    raise ValueError("SQLite dosyası bulunamadı.")
                self._db = SQLiteBackend(path=name.removeprefix("file:").split("?", 1)[0])
        return self._db

    @property
    def schema(self) -> str:
        if self._schema is None:
            self._schema = self.db.get_table_info()
        return self._schema

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            configure_openai_credentials()
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                try:
                    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                        key = st.secrets["OPENAI_API_KEY"]
                        os.environ["OPENAI_API_KEY"] = key
                except Exception:
                    pass
            if not key:
                raise ValueError("OPENAI_API_KEY bulunamadı. Lütfen ortam değişkeni veya st.secrets üzerinden tanımlayın.")
            self._llm = ChatOpenAI(model=self.model_name, temperature=0, api_key=key, timeout=30, max_retries=1)
        return self._llm

    def generate_query_json(self, question: str) -> dict[str, Any]:
        """Kullanıcı sorusunu ilişkisel JSON sorgusuna çevirir."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", _QUERY_GENERATOR_SYSTEM_PROMPT),
            ("user", "Bu soruyu yapılandırılmış JSON formatına çevir: {question}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"schema": self.schema, "question": question, "dialect": self.dialect})
        raw_text = response.content.strip()

        # Markdown işaretlerini temizle
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text).strip()

        try:
            query_json = json.loads(raw_text)
            if not isinstance(query_json, dict):
                raise ValueError("Sorgu planı JSON nesnesi olmalıdır.")
            if "unsupported" in query_json:
                raise ValueError(f"Desteklenmeyen analiz: {query_json['unsupported']}")
            # Optional nulls and singleton objects are equivalent representations
            # produced by the model; SQL expressions and invalid fields stay rejected.
            for field in ("columns", "aggregates", "group_by", "filters", "order_by", "joins", "array_joins"):
                if field in query_json and query_json[field] is None:
                    query_json.pop(field)
                elif field in {"aggregates", "filters", "order_by", "joins", "array_joins"} and isinstance(query_json.get(field), dict):
                    query_json[field] = [query_json[field]]
            return query_json
        except json.JSONDecodeError as e:
            logger.error(f"LLM çıktısı JSON olarak ayrıştırılamadı: {raw_text}")
            raise ValueError(f"Geçersiz JSON formatı: {e}") from e

    @property
    def backend(self):
        backend = adapt_database(self.db)
        if backend is None:
            raise ValueError("Gerçek veri erişim katmanı bulunamadı.")
        return backend

    def catalog(self):
        return self.backend.catalog()

    def read(self, sql, parameters=(), max_rows=1000):
        return self.backend.read(sql, parameters, max_rows)

    def periods(self, table):
        backend = self.backend
        return backend.periods(table) if backend.dialect == "clickhouse" else backend.catalog()[table]["periods"]

    def execute_nl_query(self, question: str, dialect: Optional[str] = None) -> dict[str, Any]:
        if dialect is not None and dialect != self.dialect:
            raise ValueError("Sorgu lehçesi bağlı veritabanıyla eşleşmelidir.")
        return self.execute_plan(self.generate_query_json(question), question)

    def execute_plan(self, query_json, question=""):
        from agents.results import build_result
        started = time.perf_counter()
        query_json = copy.deepcopy(query_json)
        validate_plan(query_json)
        backend = adapt_database(self.db)
        if backend is None:
            sql = compile_json_to_sql(query_json, dialect=self.dialect)
            return {"question": question, "json_query": query_json, "sql": sql, "result": self.db.run(sql)}
        validate_query_schema(query_json, backend.catalog())
        row_limit = query_json.get("limit", DEFAULT_ROWS)
        executed = copy.deepcopy(query_json)
        executed.pop("limit", None)
        parameters = []
        sql = compile_json_to_sql(executed, dialect=backend.dialect, parameters=parameters) + f" LIMIT {row_limit+1}"
        columns, rows, truncated = backend.read(sql, parameters, row_limit)
        result = build_result(question, query_json, sql, parameters, columns, rows, truncated)
        result["dialect"] = backend.dialect
        logger.info("query_completed", extra={"dialect": backend.dialect, "row_count":len(rows),
                    "duration_ms": round((time.perf_counter()-started)*1000,2), "status":result["status"]})
        return result


def get_query_agent(db_uri: str = "sqlite:///insight_generation_bot.db", model_name: str = "gpt-4o", dialect: str = "sqlite") -> QueryAgent:
    """Kolay erişim için fabrika fonksiyonu."""
    return QueryAgent(db_uri=db_uri, model_name=model_name, dialect=dialect)