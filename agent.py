import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
import streamlit as st

from agents.query_agent import QueryAgent
from agents.rewrite_nl_agent import RewriteNLAgent
from agents.data_schema import DATA_CONTEXT
from agents.credentials import configure_openai_credentials

try:
    from agents.domain_rules import (
        get_domain_context_prompt,
        build_hypothesis_synthesis_prompt,
        MARKETING_CONCEPT_DEFINITIONS,
        BUSINESS_HEURISTIC_RULES,
    )
except ImportError:
    def get_domain_context_prompt() -> str:
        return DATA_CONTEXT
    def build_hypothesis_synthesis_prompt(h: str, e: str) -> str:
        return f"Hipotez: {h}\nKanıtlar: {e}\nLütfen hipotezi doğrula ve açıkla."

logger = logging.getLogger(__name__)

# --- API ANAHTARLARI ---
try:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if hasattr(st, "secrets") and "PINECONE_API_KEY" in st.secrets:
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
except Exception:
    pass

SQL_AGENT_PREFIX = f"""
Sen üst düzey bir Pazarlama Veri Analisti ve SQL Danışmanısın.
Görevlerin:
1. VERİ SÖZLÜĞÜ:
{DATA_CONTEXT}
2. KÖK SEBEP: Konu ve demografi dağılımlarını betimle; veriyle kanıtlanmayan ürün veya nedensellik iddiaları üretme.
3. KAVRAMSAL KURALLAR:
{get_domain_context_prompt()}
"""

# --- SENTEZ MOTORU ---
# agent.py dosyasındaki SynthesisEngine sınıfını şu şekilde güncelleyin:

class SynthesisEngine:
    def __init__(self, llm_instance: ChatOpenAI):
        self.llm = llm_instance
        self.guarded_llm = RunnableLambda(lambda value: self.llm.invoke([
            SystemMessage(content=get_domain_context_prompt()),
            HumanMessage(content=value.to_string())]))

    def synthesize_executive_summary(self, question: str, sql_evidence: str) -> str:
        prompt = PromptTemplate.from_template(
            "Sen kıdemli bir Pazarlama Direktörüsün (CMO).\n"
            "Soru: {question}\n\n"
            "Veritabanından Toplanan Kanıtlar:\n{evidence}\n\n"
            "{domain_rules}\n\n"
            "GÖREVİN:\n"
            "1. Mevcut topic_name konu dağılımını ve demografik eğilimleri betimle; ürün bilgisi veya kanıtlanmamış kök neden uydurma.\n"
            "2. En fazla 3-4 cümlelik vurucu, profesyonel bir Yönetici Özeti (Final Insight) oluştur.\n"
            "3. En sona yönetici için 1 adet somut stratejik aksiyon adımı ekle."
        )
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
        prompt = PromptTemplate.from_template(
            "Hipotezler: H0={h0}; H1={h1}; H2={h2}.\nKanıtlar: {evidence}\n"
            "{domain_rules}\nHer hipotezi kanıt ve sınırlama tablosuyla değerlendir. "
            "Hesaplanmış istatistiksel test yoksa yalnızca betimsel değerlendirme yap. "
            "Destek yüzdesi veya kazanan hipotez uydurma. Sonuçları Türkçe yaz."
        )
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
        prompt = PromptTemplate.from_template(
            "Sen bir Tahminleme ve Büyüme Stratejistisin.\n"
            "Konu / Hedef: {topic}\n\n"
            "Dönemsel Zaman Serisi Verileri:\n{evidence}\n\n"
            "GÖREVİN:\n"
            "1. Geçmiş trendlerin yönünü açıkla.\n"
            "2. Bu yalnızca nitel senaryodur; sayısal tahmin veya güven aralığı uydurma.\n"
            "3. Olası riski bertaraf etmek için 2 maddelik proaktif strateji öner."
        )
        chain = prompt | self.guarded_llm
        return chain.invoke({
            "topic": topic,
            "evidence": time_series_evidence
        }).content.strip()
    
# --- HİBRİT VE KADEMELİ MOTOR ---
def get_hybrid_agent(
    db_uri: str = "sqlite:///insight_generation_bot.db",
    fast_model: str = "gpt-4o-mini",
    reasoning_model: str = "gpt-4o"
):
    configure_openai_credentials()
    db = QueryAgent(db_uri=db_uri).db
    
    # 1. Kademe: SQL ve sorgu planlayıcı model
    
    # 2. Kademe: Stratejik sentez ve hipotez doğrulama modeli
    llm_reasoning = ChatOpenAI(model=reasoning_model, temperature=0, timeout=30, max_retries=1)
    
    # Retrieval is lazy and goes through AnalysisService, not an unrestricted SQL executor.
    from agents.analysis_service import DocumentSearch
    agent_executor = DocumentSearch()

    query_agent = QueryAgent(db_uri=db_uri, model_name=fast_model)
    rewrite_agent = RewriteNLAgent(model_name=fast_model)
    synthesis_engine = SynthesisEngine(llm_reasoning)
    
    # 6 elemanı tam olarak döndürür
    return db, llm_reasoning, agent_executor, query_agent, rewrite_agent, synthesis_engine
