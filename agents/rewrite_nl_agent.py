import os
import json
import re
from typing import List, Tuple, Optional, Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from agents.data_schema import DATA_CONTEXT
from agents.credentials import configure_openai_credentials


class RewriteNLAgent:
    """
    Doğal Dil Yeniden Yazma, Çok Turlu Bağlam Takibi (Threading),
    Niyet Netleştirme ve Ayrıştırma Ajanı.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0, llm: Optional[Any] = None):
        self.model_name = model_name
        self.temperature = temperature
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            configure_openai_credentials()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                try:
                    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                        api_key = st.secrets["OPENAI_API_KEY"]
                        os.environ["OPENAI_API_KEY"] = api_key
                except Exception:
                    pass
            self._llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        return self._llm

    def contextualize_query(self, current_question: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Kullanıcının takip sorularını (Örn: 'Peki bu durum kadınlar arasında nasıl?') 
        önceki konuşma geçmişindeki filtreleri (metrik, huni aşaması, kategori) kaybetmeden 
        bağımsız, net bir SQL araştırma sorusuna dönüştürür.
        """
        if not chat_history:
            return current_question

        # Son 3 etkileşimi bağlam olarak al
        history_text = ""
        for item in chat_history[-3:]:
            history_text += f"Kullanıcı: {item.get('user', '')}\nAsistan Bulgusu: {item.get('assistant_summary', '')}\n---\n"

        context_prompt = PromptTemplate.from_template(
            "Sen bir Konuşma Bağlamı ve Takip Sorusu Çözümleyicisisin.\n\n"
            "ÖNCEKİ KONUŞMA GEÇMİŞİ:\n{history}\n\n"
            "KULLANICININ YENİ SORUSU: \"{question}\"\n\n"
            "GÖREVİN:\n"
            "1. Kullanıcının sorusu önceki konuya atıfta bulunan bir takip sorusu mu? (Örn: 'Peki kadınlar arasında nasıl?', 'Bunun sebebi ne?', 'Son çeyrekte durum ne?').\n"
            "2. Eğer takip sorusuysa, önceki konuşmada geçen konu, tüketici hunisi aşaması (Consideration, Purchase, Recommendation vb.) ve kısıtları koruyarak soruyu tam, bağımsız ve veritabanından veri çekebilecek tek bir açık soruya dönüştür.\n"
            "3. Eğer kullanıcı önceki konuyu tamamen bırakıp yeni ve bağımsız bir soru soruyorsa, soruyu hiç değiştirmeden aynen bırak.\n\n"
            "SADECE netleştirilmiş tek bir soru cümlesi yaz. Başka hiçbir açıklama ekleme."
        )

        resolved_q = (context_prompt | self.llm).invoke({
            "history": history_text.strip(),
            "question": current_question
        }).content.strip()

        return resolved_q if resolved_q else current_question

    def assess_clarification_need(self, question: str, schema: str) -> Dict[str, Any]:
        prompt = PromptTemplate.from_template(
            "Sen bir Veri Analitiği Niyet Belirleme Uzmanısın.\n"
            "Veritabanı Şeması:\n{schema}\n\n"
            "Veri sözlüğü:\n{data_context}\n\n"
            "Kullanıcı Sorusu: \"{question}\"\n\n"
            "GÖREVİN:\n"
            "Sorunun doğrudan hedeflenebilir bir odağı olup olmadığını değerlendir.\n"
            "- Net bir metrik, aşama veya demografi varsa netleştirme GEREKMEZ (needs_clarification: false).\n"
            "- Soru çok genel veya muğlaksa netleştirme GEREKİR (needs_clarification: true).\n\n"
            "YALNIZCA AŞAĞIDAKİ JSON FORMATINDA YANIT VER:\n"
            "{{\n"
            '  "needs_clarification": true,\n'
            '  "clarification_message": "Analizi daha isabetli hale getirmek için hangi ürün veya odak alanına yoğunlaşmak istersiniz?",\n'
            '  "options": [\n'
            '    {{"label": "👥 Yaş Grupları", "context": "Yaş gruplarına göre dağılıma odaklan"}},\n'
            '    {{"label": "🧭 Yolculuk Aşamaları", "context": "Yolculuk aşamalarının dağılımına odaklan"}},\n'
            '    {{"label": "💬 Konular", "context": "Konu başlıklarına göre farklı tweet sayılarına odaklan"}},\n'
            '    {{"label": "🌐 Genel Analiz", "context": "Mevcut verinin genel dağılımını analiz et"}}\n'
            '  ]\n'
            "}}\n"
            "Soru zaten netse 'needs_clarification': false ve 'options': [] döndür."
        )

        resp = (prompt | self.llm).invoke({"question": question, "schema": schema, "data_context": DATA_CONTEXT}).content.strip()
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(resp)
        except Exception:
            return {"needs_clarification": False, "options": []}

    def generate_macro_question(self, database_summary_info: str) -> str:
        hl_prompt = PromptTemplate.from_template(
            "Sen uzman bir pazarlama direktörüsün. Veritabanı özeti:\n{info}\n\n"
            "Veri sözlüğü:\n{data_context}\n\n"
            "Tüketici yolculuğundaki tıkanıklıkları sorgulayan tek bir stratejik soru üret. Sadece soruyu yaz."
        )
        return (hl_prompt | self.llm).invoke({"info": database_summary_info, "data_context": DATA_CONTEXT}).content.strip()

    def decompose_question(self, macro_question: str, schema: str) -> Tuple[str, List[str]]:
        ll_prompt = PromptTemplate.from_template(
            "Sen kıdemli bir veri analistisin. Veritabanı şeması:\n{schema}\n\n"
            "Veri sözlüğü:\n{data_context}\n\n"
            "Soru: {question}\n\n"
            "GÖREVİN: Bu soruyu çözecek 2 net alt soru üret.\n"
            "1. Yalnızca mevcut alanları ve veri sözlüğündeki ilişkileri kullan.\n"
            "2. Konu dağılımı ile kanıtlanmış kök nedeni ayır; olmayan ürün/bot filtreleri isteme.\n"
            "Soruların başına tire (-) koy."
        )
        raw_text = (ll_prompt | self.llm).invoke({"question": macro_question, "schema": schema, "data_context": DATA_CONTEXT}).content
        sub_questions = [
            line.lstrip("-* ").strip()
            for line in raw_text.split('\n')
            if line.strip().startswith(('-', '*')) and line.lstrip("-* ").strip()
        ]
        return raw_text, sub_questions

    def formulate_competing_hypotheses(self, topic: str, schema: str) -> Tuple[Dict[str, str], List[str]]:
        json_prompt = PromptTemplate.from_template(
            "Sen kıdemli bir ekonometrist ve pazarlama veri bilimcisisin.\n"
            "VERİTABANI ŞEMASI:\n{schema}\n\n"
            "Veri sözlüğü:\n{data_context}\n\n"
            "Kullanıcının Test Etmek İstediği Gözlem: {topic}\n\n"
            "Hipotezleri kullanıcının konusuna göre oluştur. Dönem kapsamını kontrol eden bir soru ekle; yıl uydurma.\n"
            "YALNIZCA AŞAĞIDAKİ GEÇERLİ JSON FORMATINDA YANIT VER:\n"
            "{{\n"
            '  "H0": "<konuya uygun sıfır hipotezi>",\n'
            '  "H1": "<kullanıcının hipotezi>",\n'
            '  "H2": "<alternatif açıklama>",\n'
            '  "test_questions": [\n'
            '    "<ilgili tabloda mevcut prediction_month değerlerini ve hacimlerini getir>",\n'
            '    "<konuya uygun ve mevcut sütunlarla cevaplanabilir ayırt edici soru>"\n'
            '  ]\n'
            "}}"
        )
        raw_resp = (json_prompt | self.llm).invoke({"topic": topic, "schema": schema, "data_context": DATA_CONTEXT}).content.strip()
        if "```json" in raw_resp:
            raw_resp = raw_resp.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_resp:
            raw_resp = raw_resp.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(raw_resp)
            hypotheses = {
                "H0": str(data.get("H0", "")).strip(),
                "H1": str(data.get("H1", "")).strip(),
                "H2": str(data.get("H2", "")).strip()
            }
            test_questions = [str(q).strip() for q in data.get("test_questions", []) if str(q).strip()]
            if len(test_questions) < 2:
                raise ValueError("Yetersiz alt soru")
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError("Hipotez planı geçersiz; konuya uygun JSON ve en az iki test sorusu gerekli.") from e

        return hypotheses, test_questions

    def formulate_hypothesis(self, topic: str, schema: str) -> Tuple[str, List[str]]:
        hyp_dict, sub_qs = self.formulate_competing_hypotheses(topic, schema)
        return f"H0: {hyp_dict.get('H0')}\nH1: {hyp_dict.get('H1')}\nH2: {hyp_dict.get('H2')}", sub_qs

    def decompose_predictive_trends(self, topic: str, schema: str) -> Tuple[str, List[str]]:
        pred_prompt = PromptTemplate.from_template(
            "Sen bir tahminleme veri bilimcisisin. Veritabanı şeması:\n{schema}\n\n"
            "Veri sözlüğü:\n{data_context}\n\n"
            "Tahmin Talebi: {question}\n\n"
            "Geçmiş trendleri verecek 2 net SQL alt sorusu kurgula. Soruların başına tire (-) koy."
        )
        raw_text = (pred_prompt | self.llm).invoke({"question": topic, "schema": schema, "data_context": DATA_CONTEXT}).content
        sub_questions = [
            line.lstrip("-* ").strip()
            for line in raw_text.split('\n')
            if line.strip().startswith(('-', '*')) and line.lstrip("-* ").strip()
        ]
        return raw_text, sub_questions


def get_rewrite_agent() -> RewriteNLAgent:
    return RewriteNLAgent()
