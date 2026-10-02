import os
import json
import re
from typing import List, Tuple, Optional, Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from agents.credentials import configure_openai_credentials

from agents.prompts.rewrite_prompts import (
    CONTEXTUALIZE_QUERY_PROMPT,
    ASSESS_CLARIFICATION_NEED_PROMPT,
    MACRO_QUESTION_PROMPT,
    DECOMPOSE_QUESTION_PROMPT,
    COMPETING_HYPOTHESES_PROMPT,
    PREDICTIVE_TRENDS_PROMPT,
)


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
            self._llm = ChatOpenAI(model=self.model_name, temperature=self.temperature, timeout=30, max_retries=1)
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
            history_text += json.dumps({"user":item.get("user", ""),"resolved_query":item.get("resolved_query", ""),"plans":item.get("context", []),"summary":item.get("assistant_summary", "")}, ensure_ascii=False) + "\n"

        context_prompt = PromptTemplate.from_template(CONTEXTUALIZE_QUERY_PROMPT)

        resolved_q = (context_prompt | self.llm).invoke({
            "history": history_text.strip(),
            "question": current_question
        }).content.strip()

        return resolved_q if resolved_q else current_question

    def route_question(self, question):
        prompt = PromptTemplate.from_template("Soruyu kaynak türüne göre sınıflandır. Yalnızca sql, documents veya hybrid yaz. Sayım/istatistik sql; PDF/politika/belge documents; ikisini isteyen hybrid. Soru: {question}")
        route = (prompt | self.llm).invoke({"question":question}).content.strip().lower()
        if route not in {"sql", "documents", "hybrid"}:
            raise ValueError("Geçerli kaynak yönlendirmesi üretilemedi.")
        return route

    def assess_clarification_need(self, question: str, schema: str) -> Dict[str, Any]:
        prompt = PromptTemplate.from_template(ASSESS_CLARIFICATION_NEED_PROMPT)

        resp = (prompt | self.llm).invoke({"question": question, "schema": schema}).content.strip()
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(resp)
        except Exception:
            return {"needs_clarification": False, "options": []}

    def generate_macro_question(self, database_summary_info: str) -> str:
        hl_prompt = PromptTemplate.from_template(MACRO_QUESTION_PROMPT)
        return (hl_prompt | self.llm).invoke({"info": database_summary_info}).content.strip()

    def decompose_question(self, macro_question: str, schema: str) -> Tuple[str, List[str]]:
        ll_prompt = PromptTemplate.from_template(DECOMPOSE_QUESTION_PROMPT)
        raw_text = (ll_prompt | self.llm).invoke({"question": macro_question, "schema": schema}).content
        sub_questions = [
            line.lstrip("-* ").strip()
            for line in raw_text.split('\n')
            if line.strip().startswith(('-', '*')) and line.lstrip("-* ").strip()
        ]
        return raw_text, sub_questions

    def formulate_competing_hypotheses(self, topic: str, schema: str) -> Tuple[Dict[str, str], List[str]]:
        json_prompt = PromptTemplate.from_template(COMPETING_HYPOTHESES_PROMPT)
        raw_resp = (json_prompt | self.llm).invoke({"topic": topic, "schema": schema}).content.strip()
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
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Hipotez planı geçersiz; konuya uygun JSON ve iki test sorusu gerekli.") from exc

        return hypotheses, test_questions

    def formulate_hypothesis(self, topic: str, schema: str) -> Tuple[str, List[str]]:
        hyp_dict, sub_qs = self.formulate_competing_hypotheses(topic, schema)
        return f"H0: {hyp_dict.get('H0')}\nH1: {hyp_dict.get('H1')}\nH2: {hyp_dict.get('H2')}", sub_qs

    def decompose_predictive_trends(self, topic: str, schema: str) -> Tuple[str, List[str]]:
        pred_prompt = PromptTemplate.from_template(PREDICTIVE_TRENDS_PROMPT)
        raw_text = (pred_prompt | self.llm).invoke({"question": topic, "schema": schema}).content
        sub_questions = [
            line.lstrip("-* ").strip()
            for line in raw_text.split('\n')
            if line.strip().startswith(('-', '*')) and line.lstrip("-* ").strip()
        ]
        return raw_text, sub_questions


def get_rewrite_agent() -> RewriteNLAgent:
    return RewriteNLAgent()