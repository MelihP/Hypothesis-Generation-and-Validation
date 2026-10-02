"""
Sorgu Planlama Ajanı ve Deterministik JSON-SQL Derleyicisi (Query Agent & JSON-to-SQL Compiler)
Doğal dil sorgularını yapılandırılmış JSON formatına dönüştürür ve ilişkisel (JOIN destekli) güvenli SQL sorguları üretir.
"""

import os
import json
import logging
import re
from typing import Any, Optional
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from agents.data_schema import DATA_CONTEXT, validate_query_schema
from sqlalchemy import inspect
from agents.sql_safety import validate_plan, read_query, database_path, DEFAULT_ROWS
import copy
import time
from pathlib import Path
from sqlalchemy.engine import make_url
from agents.credentials import configure_openai_credentials
import streamlit as st

logger = logging.getLogger(__name__)

# --- 1. API ANAHTARLARI (Streamlit Secrets & Ortam Değişkenleri ile Uyumlu) ---
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# --- 2. OPERATÖR HARİTASI VE TOPLAMA FONKSİYONLARI ---
_FILTER_OP_TO_SQL = {
    "EQ": "=",
    "NEQ": "!=",
    "GT": ">",
    "GTE": ">=",
    "LT": "<",
    "LTE": "<=",
    "LIKE": "LIKE",
    "ILIKE": "LIKE",
    "IN": "IN",
    "NOT_IN": "NOT IN",
    "IS_NULL": "IS NULL",
    "IS_NOT_NULL": "IS NOT NULL",
    "BETWEEN": "BETWEEN",
}

_AGG_OPS = frozenset({"count", "count_distinct", "sum", "avg", "min", "max"})
_DANGEROUS_IDENT_RE = re.compile(r"[;\x00]|--|/\*")


# --- 3. GELİŞTİRİLMİŞ AJAN SİSTEM PROMPTU (AŞAMA 2 MANTIĞI) ---
_QUERY_GENERATOR_SYSTEM_PROMPT = """\
Sen uzman bir SQL ve JSON Sorgu Planlama Ajanısın (Query Planning Agent).
Görevin: Kullanıcının doğal dilde sorduğu iş veya pazarlama sorusunu inceleyerek, verilen veritabanı şemasına uygun yapılandırılmış (structured) bir JSON sorgu nesnesine (SPJQ) dönüştürmektir.

VERİTABANI ŞEMASI:
{schema}

BEKLENEN JSON ÇIKTI FORMATI:
{
  "table": "<ana_tablo_adi>",
  "joins": [
    {
      "type": "INNER" | "LEFT",
      "table": "<ikinci_tablo>",
      "on": {"left": "<tablo1.sutun>", "right": "<tablo2.sutun>"}
    }
  ],
  "columns": ["<sutun1>", "<sutun2>"],
  "aggregates": [
    {"op": "count" | "count_distinct" | "sum" | "avg" | "min" | "max", "column": "<sutun_adi>", "as": "<takma_ad>"}
  ],
  "filters": [
    {"column": "<tablo_veya_sutun_adi>", "op": "EQ" | "NEQ" | "GT" | "GTE" | "LT" | "LTE" | "LIKE" | "ILIKE" | "IN" | "NOT_IN" | "BETWEEN" | "IS_NULL" | "IS_NOT_NULL", "value": <deger>}
  ],
  "group_by": ["<sutun1>", "<sutun2>"],
  "order_by": [
    {"column": "<sutun_veya_takma_ad>", "dir": "asc" | "desc"}
  ],
  "limit": <sayi>
}

Yalnızca saf JSON formatında yanıt döndür. Markdown kod blokları veya açıklama metni KOYMA.
Sorguya uygun bir limit belirle (örneğin 10).
JOIN birden fazla koşul gerektiriyorsa on alanında left/right nesnelerinin listesini kullan.
Örnek: "on": [{"left": "a.tweet_id", "right": "b.tweet_id"}, {"left": "a.prediction_month", "right": "b.prediction_month"}].
İstenen alan şemada yoksa sorgu uydurmak yerine {"unsupported": "Türkçe gerekçe"} döndür.
"""


# --- 4. DETERMINİSTİK VE İLİŞKİSEL (JOIN DESTEKLİ) JSON-TO-SQL DERLEYİCİSİ ---

def quote_ident(name: str, quote_char: str = '"') -> str:
    """Tablo ve sütun adlarını nokta ayrımı ve fonksiyon güvenliğiyle çift tırnak içine alır."""
    if not name or _DANGEROUS_IDENT_RE.search(name):
        raise ValueError(f"Geçersiz tanımlayıcı (Identifier): {name!r}")
    
    # json_each(tablo.sutun) gibi SQLite fonksiyonlarını koru
    m = re.match(r"^json_each\((.+)\)$", name.strip(), re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        return f"json_each({quote_ident(inner, quote_char)})"

    # tablo.sutun formatındaki ilişkisel tanımlayıcıları ayrı ayrı tırnakla
    if "." in name:
        return ".".join(quote_ident(p.strip(), quote_char) for p in name.split("."))
        
    escaped = name.replace(quote_char, quote_char * 2)
    return f"{quote_char}{escaped}{quote_char}"


def _lit(v: Any) -> str:
    """Değerleri SQL injection güvenliği için kaçışlayarak biçimlendirir."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("\x00", "").replace("'", "''")
    return f"'{s}'"


def _contains_lit(v: Any) -> str:
    """LIKE / ILIKE sorguları için güvenli '%metin%' formatına dönüştürür."""
    s = str(v).replace("\x00", "").replace("'", "''")
    return f"'%{s}%'"


def _compile_join(j: dict[str, Any]) -> str:
    on = j["on"]
    conditions = on if isinstance(on, list) else [on]
    table = quote_ident(j["table"])
    if j.get("alias"):
        table += f" AS {quote_ident(j['alias'])}"
    clause = " AND ".join(f"{quote_ident(c['left'])} = {quote_ident(c['right'])}" for c in conditions)
    return f"{j.get('type', 'INNER').upper()} JOIN {table} ON {clause}"


def compile_json_to_sql(query_json: dict[str, Any], dialect: str = "sqlite", parameters: Optional[list] = None) -> str:
    """
    Yapılandırılmış JSON sorgu nesnesini deterministik ve güvenli SQL ifadesine dönüştürür.
    JOIN'ler, demografik eşleşmeler ve toplama fonksiyonları tam desteklenir.
    """
    validate_plan(query_json)
    def literal(value):
        if parameters is None:
            return _lit(value)
        parameters.append(value)
        return "?"
    table = query_json.get("table")
    if not table:
        raise ValueError("JSON sorgusunda zorunlu 'table' alanı eksik!")

    q_table = quote_ident(table)
    alias = query_json.get("alias")
    if alias:
        q_table += f" AS {quote_ident(str(alias))}"

    joins = query_json.get("joins") or []
    columns = query_json.get("columns") or []
    aggregates = query_json.get("aggregates") or []
    group_by = query_json.get("group_by") or []
    filters = query_json.get("filters") or []
    order_by = query_json.get("order_by") or []
    limit = query_json.get("limit")

    select_parts: list[str] = [quote_ident(str(g)) for g in group_by]

    for agg in aggregates:
        if not isinstance(agg, dict):
            continue
        op = (agg.get("op") or "").lower().strip()
        if op not in _AGG_OPS:
            raise ValueError(f"Desteklenmeyen toplama operatörü: {op!r}")
        col = agg.get("column")
        safe_col_name = str(col).replace(".", "_") if col else op
        alias_agg = agg.get("as") or f"{op}_{safe_col_name}"
        q_alias = quote_ident(alias_agg)

        if op == "count" and not col:
            expr = "count(*)"
        elif op == "count_distinct":
            if not col:
                raise ValueError("count_distinct işlemi için sütun adı zorunludur.")
            expr = f"count(DISTINCT {quote_ident(str(col))})"
        elif op == "count":
            expr = f"count({quote_ident(str(col))})"
        else:
            if not col:
                raise ValueError(f"'{op}' toplama işlemi için sütun adı zorunludur.")
            expr = f"{op}({quote_ident(str(col))})"

        select_parts.append(f"{expr} AS {q_alias}")

    if not select_parts:
        select_parts = [quote_ident(str(c)) for c in columns] if columns else ["*"]

    select_clause = ", ".join(select_parts)
    sql = f"SELECT {select_clause} FROM {q_table}"

    # JOIN ifadelerini ekle
    for j in joins:
        j_sql = _compile_join(j)
        if j_sql:
            sql += f" {j_sql}"

    # WHERE koşullarını derle
    where_parts: list[str] = []
    for f in filters:
        if not isinstance(f, dict):
            continue
        col = f.get("column")
        op_key = (f.get("op") or "").upper().strip()
        sql_op = _FILTER_OP_TO_SQL.get(op_key)
        if not col or sql_op is None:
            continue
        q_col = quote_ident(str(col))
        val = f.get("value")

        if sql_op in ("IS NULL", "IS NOT NULL"):
            where_parts.append(f"{q_col} {sql_op}")
        elif sql_op in ("IN", "NOT IN"):
            if isinstance(val, dict) and "table" in val:
                subquery_dict = dict(val)
                if not subquery_dict.get("columns") and subquery_dict.get("column"):
                    subquery_dict["columns"] = [subquery_dict.pop("column")]
                sub_sql = compile_json_to_sql(subquery_dict, dialect=dialect, parameters=parameters)
                where_parts.append(f"{q_col} {sql_op} ({sub_sql})")
            elif isinstance(val, (list, tuple)) and len(val) == 1 and isinstance(val[0], dict) and "table" in val[0]:
                subquery_dict = dict(val[0])
                if not subquery_dict.get("columns") and subquery_dict.get("column"):
                    subquery_dict["columns"] = [subquery_dict.pop("column")]
                sub_sql = compile_json_to_sql(subquery_dict, dialect=dialect, parameters=parameters)
                where_parts.append(f"{q_col} {sql_op} ({sub_sql})")
            else:
                vals = val if isinstance(val, (list, tuple)) else [val]
                if not vals:
                    where_parts.append("0 = 1" if sql_op == "IN" else "1 = 1")
                else:
                    where_parts.append(f"{q_col} {sql_op} ({', '.join(literal(v) for v in vals)})")
        elif sql_op == "BETWEEN":
            if isinstance(val, (list, tuple)) and len(val) == 2:
                where_parts.append(f"{q_col} BETWEEN {literal(val[0])} AND {literal(val[1])}")
        elif op_key in ("LIKE", "ILIKE"):
            where_parts.append(f"{q_col} LIKE {literal(chr(37) + str(val) + chr(37))}")
        elif val is None and op_key in ("EQ", "NEQ"):
            where_parts.append(f"{q_col} IS {'NOT ' if op_key == 'NEQ' else ''}NULL")
        else:
            where_parts.append(f"{q_col} {sql_op} {literal(val)}")

    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    if group_by:
        sql += " GROUP BY " + ", ".join(quote_ident(str(g)) for g in group_by)

    order_parts: list[str] = []
    for o in order_by:
        if not isinstance(o, dict):
            continue
        col = o.get("column")
        if not col:
            continue
        direction = "DESC" if str(o.get("dir", "")).lower() == "desc" else "ASC"
        order_parts.append(f"{quote_ident(str(col))} {direction}")

    if order_parts:
        sql += " ORDER BY " + ", ".join(order_parts)

    if limit is not None:
        try:
            sql += f" LIMIT {int(limit)}"
        except (ValueError, TypeError):
            pass

    return sql


# --- 5. SORGU PLANLAMA AJANI SINIFI (Lazy-Loaded LLM & DB) ---

class QueryAgent:
    """
    Doğal dil sorularını yapılandırılmış JSON formatına çeviren ve deterministik SQL üreten sorgu ajanı.
    """
    def __init__(
        self,
        db_uri: str = "sqlite:///insight_generation_bot.db",
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        llm: Optional[Any] = None,
        db: Optional[Any] = None,
        schema: Optional[str] = None
    ):
        self.db_uri = db_uri
        self.model_name = model_name
        self.api_key = api_key
        self._db = db
        self._llm = llm
        self._schema = schema

    @property
    def db(self) -> SQLDatabase:
        if self._db is None:
            url = make_url(self.db_uri)
            if url.get_backend_name() != "sqlite" or not url.database:
                raise ValueError("Dosya tabanlı SQLite gereklidir.")
            name = url.database.removeprefix("file:").split("?", 1)[0]
            path = Path(name).resolve()
            if not path.is_file():
                raise ValueError("SQLite veritabanı bulunamadı.")
            self._db = SQLDatabase.from_uri(f"sqlite:///file:{path}?mode=ro&uri=true", sample_rows_in_table_info=0)
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
            # A concrete message preserves literal JSON braces and schema content.
            SystemMessage(content=_QUERY_GENERATOR_SYSTEM_PROMPT.replace("{schema}", self.schema) + DATA_CONTEXT),
            ("user", "Bu soruyu yapılandırılmış JSON formatına çevir: {question}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"schema": self.schema, "question": question})
        raw_text = response.content.strip()

        # Markdown işaretlerini temizle
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text).strip()

        try:
            query_json = json.loads(raw_text)
            if not isinstance(query_json, dict):
                raise ValueError("Sorgu planı bir JSON nesnesi olmalıdır.")
            if "unsupported" in query_json:
                raise ValueError(f"Desteklenmeyen analiz: {query_json['unsupported']}")
            return query_json
        except json.JSONDecodeError as e:
            logger.error(f"LLM çıktısı JSON olarak ayrıştırılamadı: {raw_text}")
            raise ValueError(f"Geçersiz JSON formatı: {e}") from e

    def execute_nl_query(self, question: str) -> dict[str, Any]:
        return self.execute_plan(self.generate_query_json(question), question)

    def execute_plan(self, query_json: dict, question: str = "") -> dict[str, Any]:
        from agents.results import build_result
        started = time.monotonic()
        query_json = copy.deepcopy(query_json)
        validate_plan(query_json)
        parameters = []
        if isinstance(self.db, SQLDatabase):
            inspector = inspect(self.db._engine)
            tables = {name: {col["name"] for col in inspector.get_columns(name)}
                      for name in self.db.get_usable_table_names()}
            validate_query_schema(query_json, tables)
            # Fetch one extra row to expose a clipped result, including requested top-N.
            row_limit = query_json.get("limit", DEFAULT_ROWS)
            executed = copy.deepcopy(query_json)
            executed.pop("limit", None)
            sql = compile_json_to_sql(executed, parameters=parameters) + f" LIMIT {row_limit + 1}"
            columns, rows, truncated = read_query(database_path(self.db), sql, parameters,
                                                  max_rows=row_limit, allowed_tables=set(tables))
            result = build_result(question, query_json, sql, parameters, columns, rows, truncated)
        else:
            # Dependency-injected test adapters retain the legacy return contract.
            sql = compile_json_to_sql(query_json)
            result = {"question": question, "json_query": query_json, "sql": sql, "result": self.db.run(sql)}
        logger.info("query_completed", extra={"duration_ms": round((time.monotonic()-started)*1000, 2),
                    "row_count": len(result.get("rows", [])), "status": result.get("status", "success")})
        return result


def get_query_agent(db_uri: str = "sqlite:///insight_generation_bot.db", model_name: str = "gpt-4o") -> QueryAgent:
    """Kolay erişim için fabrika fonksiyonu."""
    return QueryAgent(db_uri=db_uri, model_name=model_name)
