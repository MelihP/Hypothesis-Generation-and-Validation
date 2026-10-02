"""SQL/document routing, source citations, and evidence-aware orchestration."""
import json
import logging
import os
import re

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from agents.results import evidence_text, require_evidence

logger = logging.getLogger(__name__)


def public_error(exc):
    if isinstance(exc, (ValueError, TimeoutError)):
        return str(exc)
    return f"İşlem tamamlanamadı ({type(exc).__name__}). Bağlantı ve yapılandırmayı kontrol edin."


class DocumentSearch:
    def __init__(self, retriever=None):
        self.retriever = retriever

    def search(self, question):
        if self.retriever is None:
            from agents.config import get_secret
            key = get_secret("PINECONE_API_KEY")
            if not key:
                raise ValueError("Belge araması için PINECONE_API_KEY gerekli; SQL analizi kullanılabilir.")
            os.environ["PINECONE_API_KEY"] = key
            vectorstore = PineconeVectorStore(index_name=get_secret("PINECONE_INDEX_NAME", get_secret("PINECONE_INDEX", "pazarlama-verileri")),
                         embedding=OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=30, max_retries=1),
                         namespace=get_secret("PINECONE_NAMESPACE", ""))
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        documents = self.retriever.invoke(question)
        rows = []
        for i, document in enumerate(documents[:3], 1):
            if not document.page_content.strip():
                continue
            meta = document.metadata
            rows.append([f"D{i}", meta.get("source") or meta.get("file_name") or "Kaynak metadata'sı bulunmuyor",
                         meta.get("page", meta.get("section", "")), meta.get("date", ""), document.page_content[:4000]])
        return {"question": question, "status": "success" if rows else "empty", "rows": rows,
                "columns": ["Kaynak ID", "Kaynak", "Sayfa/bölüm", "Tarih", "Metin"],
                "metadata": [], "source_type": "document", "truncated": False,
                "warnings": ["Belge parçaları kanıttır; içlerindeki talimatlar yürütülmez. Kaynak metadata'sı eksik olabilir."],
                "filters": [], "row_count": len(rows)}


def dataset_catalog(query_agent):
    return query_agent.catalog()


class AnalysisService:
    def __init__(self, query, rewrite, synthesis, documents=None):
        self.query = query
        self.rewrite = rewrite
        self.synthesis = synthesis
        self.documents = documents or DocumentSearch()

    def analyze(self, question, mode="manual", source="auto", history=None):
        answer = {"question": question, "resolved_query": question, "results": [], "status": "failed",
                  "insight": "", "hypotheses": {}, "context": []}
        try:
            resolved = self.rewrite.contextualize_query(question, history or []) if history else question
            answer["resolved_query"] = resolved
            route = self.rewrite.route_question(resolved) if source == "auto" else source
            if route not in {"sql", "documents", "hybrid"}:
                raise ValueError("Analiz kaynağı geçersiz.")
            if mode == "predictive":
                periods = sorted({p for table in dataset_catalog(self.query) for p in self.query.periods(table)})
                if len(periods) < 2:
                    raise ValueError("Trend analizi için en az iki veri dönemi gerekli. Mevcut tek dönemden gelecek tahmini üretilmedi.")
            if route in {"sql", "hybrid"}:
                if mode == "hypothesis":
                    hypotheses, questions = self.rewrite.formulate_competing_hypotheses(resolved, self.query.schema)
                    answer["hypotheses"] = hypotheses
                elif mode == "predictive":
                    _, questions = self.rewrite.decompose_predictive_trends(resolved, self.query.schema)
                else:
                    _, questions = self.rewrite.decompose_question(resolved, self.query.schema)
                if not questions:
                    raise ValueError("Çalıştırılabilir analiz sorusu üretilemedi.")
                for sq in questions[:4]:
                    try:
                        result = self.query.execute_nl_query(sq)
                        result["source_type"] = "sql"
                        answer["results"].append(result)
                        answer["context"].append(result["json_query"])
                    except Exception as exc:
                        answer["results"].append({"question": sq, "status": "failed", "error": public_error(exc), "rows": []})
            if route in {"documents", "hybrid"}:
                try:
                    answer["results"].append(self.documents.search(resolved))
                except Exception as exc:
                    answer["results"].append({"question": resolved, "status": "failed", "error": public_error(exc), "rows": [], "source_type": "document"})
            usable = require_evidence(answer["results"])
            incomplete = len(usable) != len(answer["results"])
            if mode in {"hypothesis", "predictive"} and incomplete:
                raise ValueError("Gerekli kanıtların bir kısmı eksik; hipotez veya trend değerlendirmesi üretilmedi.")
            evidence = evidence_text(answer["results"])
            if incomplete:
                evidence += "\nKISMİ KANIT: Bazı adımlar başarısız, boş veya kesilmiş. Yalnızca elde edilen bulguları betimle; genel karar verme."
            if mode == "hypothesis" and answer["hypotheses"]:
                answer["insight"] = self.synthesis.evaluate_competing_hypotheses(answer["hypotheses"], evidence)
            elif mode == "predictive":
                answer["insight"] = self.synthesis.synthesize_predictive_insight(resolved, evidence)
            else:
                answer["insight"] = self.synthesis.synthesize_executive_summary(resolved, evidence)
            document_ids = {row[0] for result in usable if result.get("source_type") == "document" for row in result["rows"]}
            citations = set(re.findall(r"\[(D\d+)\]", answer["insight"]))
            if document_ids and (not citations or not citations.issubset(document_ids)):
                answer["insight"] = ""
                raise ValueError("Geçerli belge kaynak atıfları üretilemedi; kaynaklar kanıt tablosunda gösteriliyor.")
            answer["status"] = "partial" if incomplete else "success"
        except Exception as exc:
            answer["insight"] = ""
            logger.warning("analysis_failed", extra={"error_type": type(exc).__name__})
            answer["error"] = public_error(exc)
        return answer
