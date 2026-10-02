import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from agents.credentials import configure_openai_credentials
from agents.config import get_secret, enabled
from agents.database import SQLiteBackend, ClickHouseBackend
from pathlib import Path
import streamlit as st

from agents.query_agent import QueryAgent, CLICKHOUSE_CORE_TABLES
from agents.rewrite_nl_agent import RewriteNLAgent
from logger import logger, log_db_fallback

try:
    from agents.prompts.domain_prompts import (
        get_domain_context_prompt,
        build_hypothesis_synthesis_prompt,
        MARKETING_CONCEPT_DEFINITIONS,
        BUSINESS_HEURISTIC_RULES,
    )
except ImportError:
    def get_domain_context_prompt() -> str:
        return "Pazarlama hunisinde Consideration düşüşü funnel daralmasını gösterir. Kök neden için topics/products sütunlarına odaklan."
    def build_hypothesis_synthesis_prompt(h: str, e: str) -> str:
        return f"Hipotez: {h}\nKanıtlar: {e}\nLütfen hipotezi doğrula ve açıkla."


# --- 1. API ANAHTARLARI & ORTAM DEĞİŞKENLERİ ---
for key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "OPENAI_MODEL_NAME"]:
    val = get_secret(key)
    if val:
        os.environ[key] = val


from agents.prompts.query_prompts import SQL_AGENT_PREFIX
from agents.prompts.synthesis_prompts import (
    EXECUTIVE_SUMMARY_PROMPT,
    COMPETING_HYPOTHESES_EVALUATION_PROMPT,
    PREDICTIVE_INSIGHT_PROMPT,
)


# --- 2. SENTEZ MOTORU (SYNTHESIS ENGINE) ---

class SynthesisEngine:
    def __init__(self, llm_instance: ChatOpenAI):
        self.llm = llm_instance
        self.guarded_llm = RunnableLambda(lambda value: self.llm.invoke([
            SystemMessage(content=get_domain_context_prompt()), HumanMessage(content=value.to_string())]))

    def synthesize_executive_summary(self, question: str, sql_evidence: str) -> str:
        prompt = PromptTemplate.from_template(EXECUTIVE_SUMMARY_PROMPT)
        chain = prompt | self.guarded_llm
        return chain.invoke({
            "question": question,
            "evidence": sql_evidence,
            "domain_rules": get_domain_context_prompt()
        }).content.strip()

    def evaluate_competing_hypotheses(self, hypotheses: Dict[str, str], sql_evidence: str) -> str:
        """
        H0, H1 ve H2 hipotezlerini toplanan SQL verisi karşısında eşzamanlı yarıştırır.
        Varsayımsal konuşmaz; verideki reel sayıları kanıt göstererek karne üretir.
        """
        prompt = PromptTemplate.from_template(COMPETING_HYPOTHESES_EVALUATION_PROMPT)
        chain = prompt | self.guarded_llm
        return chain.invoke({
            "h0": hypotheses.get("H0", "Sıfır hipotezi"),
            "h1": hypotheses.get("H1", "Birincil hipotez"),
            "h2": hypotheses.get("H2", "Rakip hipotez"),
            "evidence": sql_evidence,
            "domain_rules": get_domain_context_prompt()
        }).content.strip()

    def verify_hypothesis(self, hypothesis: str, sql_evidence: str) -> str:
        prompt_text = build_hypothesis_synthesis_prompt(hypothesis, sql_evidence)
        response = self.llm.invoke([SystemMessage(content=get_domain_context_prompt()), HumanMessage(content=prompt_text)])
        return response.content.strip()

    def synthesize_predictive_insight(self, topic: str, time_series_evidence: str) -> str:
        prompt = PromptTemplate.from_template(PREDICTIVE_INSIGHT_PROMPT)
        chain = prompt | self.guarded_llm
        return chain.invoke({
            "topic": topic,
            "evidence": time_series_evidence
        }).content.strip()


# --- 3. VERİTABANI BAĞLANTISI VE YEDEKLEME (CLICKHOUSE -> SQLITE FALLBACK) ---

def get_database_connection(custom_uri=None):
    """Configured ClickHouse never silently turns into an unrelated SQLite dataset."""
    from sqlalchemy.engine import make_url
    import clickhouse_connect
    local_path = Path(__file__).resolve().parent / "insight_generation_bot.db"
    if custom_uri and "clickhouse" not in custom_uri.lower():
        name = make_url(custom_uri).database
        if not name:
            raise ValueError("SQLite dosyası bulunamadı.")
        db = SQLiteBackend(path=name.removeprefix("file:").split("?",1)[0])
        return db, custom_uri, "sqlite"
    host = get_secret("CLICKHOUSE_HOST")
    if custom_uri or host:
        if custom_uri:
            url = make_url(custom_uri)
            host = url.host
            port = url.port or 8123
            username = url.username or "default"
            password = url.password or ""
            database = url.database or "default"
            secure = str(url.query.get("secure", port in (443,8443))).lower() in {"true","1","yes"}
            verify = str(url.query.get("verify", "true")).lower() not in {"false","0","no"}
        else:
            secure = enabled("CLICKHOUSE_SECURE", get_secret("CLICKHOUSE_PORT") in {"443", "8443"})
            port = int(get_secret("CLICKHOUSE_PORT", "8443" if secure else "8123"))
            username = get_secret("CLICKHOUSE_USERNAME", "default")
            password = get_secret("CLICKHOUSE_PASSWORD")
            database = get_secret("CLICKHOUSE_DB", "default")
            verify = not get_secret("CLICKHOUSE_VERIFY", "true").lower() in {"false","0","no"}
        options = {"host": host, "port":port, "username":username, "password":password,
                   "database":database, "secure":secure, "verify":verify, "connect_timeout":5, "send_receive_timeout":15}
        ca_cert = get_secret("CLICKHOUSE_CA_CERT")
        if ca_cert:
            options["ca_cert"] = ca_cert
        try:
            backend = ClickHouseBackend(clickhouse_connect.get_client(**options), database)
            backend.tls_verification_disabled = secure and not verify
            backend.catalog()
            return backend, "clickhousedb://configured", "clickhouse"
        except Exception as exc:
            logger.warning("clickhouse_connection_failed", extra={"error_type":type(exc).__name__})
            if not enabled("ALLOW_SQLITE_FALLBACK"):
                raise ValueError("ClickHouse bağlantısı kurulamadı. Yapılandırma, ağ erişimi ve salt okunur hesap izinlerini kontrol edin; SQLite'a otomatik geçilmedi.") from None
            backend = SQLiteBackend(path=local_path, fallback=True)
            return backend, f"sqlite:///{local_path}", "sqlite"
    backend = SQLiteBackend(path=local_path)
    return backend, f"sqlite:///{local_path}", "sqlite"


# --- 4. HİBRİT VE KADEMELİ MOTOR (2-TIER ROUTING) ---

def get_hybrid_agent(
    db_uri: Optional[str] = None,
    fast_model: str = "gpt-4o-mini",
    reasoning_model: str = "gpt-4o"
):
    configure_openai_credentials()
    db, active_uri, dialect = get_database_connection(custom_uri=db_uri)
    llm_reasoning = ChatOpenAI(model=reasoning_model, temperature=0, timeout=30, max_retries=1)
    from agents.analysis_service import DocumentSearch
    documents = DocumentSearch()
    query_agent = QueryAgent(db_uri=active_uri, model_name=fast_model, dialect=dialect, db=db)
    rewrite_agent = RewriteNLAgent(model_name=fast_model)
    synthesis_engine = SynthesisEngine(llm_reasoning)
    return db, llm_reasoning, documents, query_agent, rewrite_agent, synthesis_engine
