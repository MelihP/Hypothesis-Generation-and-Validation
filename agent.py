import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
import streamlit as st

from agents.query_agent import QueryAgent
from agents.rewrite_nl_agent import RewriteNLAgent

try:
    from agents.domain_rules import (
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
1. KÖK SEBEP ÖNCELİĞİ: 'Neden', 'ürün problemi', 'şikayet kaynağı' gibi sorularda 'emotions' sütununu tek başına KULLANMA. 'topics', 'topic_categories' ve 'products' sütunlarındaki gerçek operasyonel sebepleri bul.
2. DEMOGRAFİK BİRLEŞTİRME (JOIN): Yaş veya cinsiyet sorulduğunda 'twitter_tweets' ile 'demo_brand_users' tablolarını 'author_id = id' üzerinden birleştir (JOIN). Botları hariç tutmak için 'is_org = 0' filtresi uygula.
3. KAVRAMSAL KURALLAR:
{get_domain_context_prompt()}
"""

# --- SENTEZ MOTORU ---
# agent.py dosyasındaki SynthesisEngine sınıfını şu şekilde güncelleyin:

class SynthesisEngine:
    def __init__(self, llm_instance: ChatOpenAI):
        self.llm = llm_instance

    def synthesize_executive_summary(self, question: str, sql_evidence: str) -> str:
        prompt = PromptTemplate.from_template(
            "Sen kıdemli bir Pazarlama Direktörüsün (CMO).\n"
            "Soru: {question}\n\n"
            "Veritabanından Toplanan Kanıtlar:\n{evidence}\n\n"
            "{domain_rules}\n\n"
            "GÖREVİN:\n"
            "1. Yalnızca duygulardan bahsetme; arka plandaki kök nedenleri (topics, products) ve demografik eğilimleri vurgula.\n"
            "2. En fazla 3-4 cümlelik vurucu, profesyonel bir Yönetici Özeti (Final Insight) oluştur.\n"
            "3. En sona yönetici için 1 adet somut stratejik aksiyon adımı ekle."
        )
        chain = prompt | self.llm
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
            "Sen Baş Ekonometrist ve Kıdemli Pazarlama Direktörüsün (CMO).\n\n"
            "YARIŞAN HİPOTEZLER:\n"
            "- H0 (Sıfır Hipotezi): {h0}\n"
            "- H1 (Birincil Hipotez): {h1}\n"
            "- H2 (Rakip Hipotez): {h2}\n\n"
            "VERİTABANINDAN TOPLANAN GERÇEK KANITLAR:\n{evidence}\n\n"
            "PAZARLAMA KURALLARI:\n{domain_rules}\n\n"
            "GÖREVİN:\n"
            "1. KESİNLİKLE VARSAYIMSAL ('Eğer yüksekse', 'varsayarsak', 'olabilir') KONUŞMA. "
            "   SQL çıktısında hangi sayılar, hacimler veya sıfırlar varsa doğrudan bu reel rakamları referans ver.\n"
            "2. KARŞILAŞTIRMALI HİPOTEZ KARNESİ (Markdown Tablosu formatında üret):\n"
            "   | Hipotez | Açıklama | Karar ([KABUL] / [KISMEN] / [REDDEDİLDİ]) | Destek Skoru (%) | Verideki Somut Kanıt (Sayılar/Metrikler) |\n"
            "3. KAZANAN HİPOTEZ VE DERİN ANALİZ: Kazanan hipotezi ilan et; funnel daralmasını ve verideki sayısal çöküşü pazarlama mantığıyla açıkla.\n"
            "4. YÖNETİCİ EYLEM PLANI: 2 maddelik net ve somut aksiyon adımı öner.\n\n"
            "Yanıtını profesyonel, net ve Türkçe olarak sun."
        )
        chain = prompt | self.llm
        return chain.invoke({
            "h0": hypotheses.get("H0", "Sıfır hipotezi"),
            "h1": hypotheses.get("H1", "Birincil hipotez"),
            "h2": hypotheses.get("H2", "Rakip hipotez"),
            "evidence": sql_evidence,
            "domain_rules": get_domain_context_prompt()
        }).content.strip()

    def verify_hypothesis(self, hypothesis: str, sql_evidence: str) -> str:
        prompt_text = build_hypothesis_synthesis_prompt(hypothesis, sql_evidence)
        response = self.llm.invoke(prompt_text)
        return response.content.strip()

    def synthesize_predictive_insight(self, topic: str, time_series_evidence: str) -> str:
        prompt = PromptTemplate.from_template(
            "Sen bir Tahminleme ve Büyüme Stratejistisin.\n"
            "Konu / Hedef: {topic}\n\n"
            "Dönemsel Zaman Serisi Verileri:\n{evidence}\n\n"
            "GÖREVİN:\n"
            "1. Geçmiş trendlerin yönünü açıkla.\n"
            "2. Gelecek dönem için risk ve fırsat projeksiyonu yap.\n"
            "3. Olası riski bertaraf etmek için 2 maddelik proaktif strateji öner."
        )
        chain = prompt | self.llm
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
    db = SQLDatabase.from_uri(db_uri)
    
    # 1. Kademe: SQL ve sorgu planlayıcı model
    llm_fast = ChatOpenAI(model=fast_model, temperature=0)
    
    # 2. Kademe: Stratejik sentez ve hipotez doğrulama modeli
    llm_reasoning = ChatOpenAI(model=reasoning_model, temperature=0)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    index_name = "pazarlama-verileri" 
    extra_tools = []
    try:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        rag_tool = Tool(
            name="dokuman_arama_araci",
            description="Markanın iade politikaları, PDF strateji raporları veya SQL veritabanında olmayan yapılandırılmamış metinleri araştırmak için bu aracı kullan.",
            func=retriever.invoke
        )
        extra_tools.append(rag_tool)
    except Exception as e:
        logger.warning(f"RAG sistemine bağlanılamadı: {e}")

    agent_executor = create_sql_agent(
        llm=llm_fast,
        db=db,
        agent_type="openai-tools",
        extra_tools=extra_tools,
        prefix=SQL_AGENT_PREFIX,
        verbose=False
    )
    
    query_agent = QueryAgent(db_uri=db_uri, model_name=fast_model)
    rewrite_agent = RewriteNLAgent(model_name=fast_model)
    synthesis_engine = SynthesisEngine(llm_reasoning)
    
    # 6 elemanı tam olarak döndürür
    return db, llm_reasoning, agent_executor, query_agent, rewrite_agent, synthesis_engine