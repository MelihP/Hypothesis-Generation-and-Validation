from pathlib import Path
import unittest
import os
from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest
from agents.results import build_result

APP = str(Path(__file__).resolve().parents[1]/"app.py")


def widget(collection, label):
    return next(w for w in collection if w.label == label)


class TestAppWorkflows(unittest.TestCase):
    def app(self):
        return AppTest.from_file(APP).run(timeout=20)

    def explorer(self, app):
        widget(app.radio, "Çalışma modu").set_value("📊 Veri Sorgulama ve İstatistik")
        return app.run(timeout=20)

    def test_all_modes_open_without_api_calls(self):
        app = self.app()
        for mode in widget(app.radio, "Çalışma modu").options:
            widget(app.radio, "Çalışma modu").set_value(mode)
            app.run(timeout=20)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get("doc_string")), 0)

    def test_builder_produces_real_table_and_downloads(self):
        app = self.explorer(self.app())
        widget(app.selectbox, "Tablo").set_value("demographics")
        app.run(timeout=20)
        widget(app.multiselect, "Gruplama sütunları (çapraz tablo için iki sütun)").set_value(["age_group"])
        widget(app.button, "Tabloyu oluştur").click()
        app.run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        result = app.session_state["answers"]["builder"]
        self.assertEqual(sum(row[1] for row in result["rows"]), 5997)
        self.assertFalse(result["truncated"])
        self.assertIn("kayıt_sayısı", result["columns"])

    def test_full_table_quality_and_real_statistics(self):
        app = self.explorer(self.app())
        widget(app.radio, "İşlem").set_value("Veri kalitesi")
        app.run(timeout=20)
        widget(app.selectbox, "Kalite kontrolü tablosu").set_value("demographics")
        app.run(timeout=20)
        widget(app.button, "Eksik veri analizini çalıştır").click()
        app.run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue((app.session_state["answers"]["quality"]["Toplam"] == 5997).all())
        widget(app.radio, "İşlem").set_value("İstatistiksel karşılaştırma")
        app.run(timeout=20)
        widget(app.selectbox, "Karşılaştırma verisi").set_value("consumer_journey")
        app.run(timeout=20)
        widget(app.selectbox, "Bağımsız analiz birimi anahtarı").set_value("author_id")
        widget(app.selectbox, "Sonuç / etiket sütunu").set_value("journey_stage")
        widget(app.selectbox, "Demografik / grup tablosu").set_value("demographics")
        app.run(timeout=20)
        widget(app.selectbox, "Grup sütunu").set_value("age_group")
        widget(app.selectbox, "Grup tablosu eşleşme anahtarı").set_value("user_id")
        widget(app.selectbox, "Karşılaştırma dönemi").set_value(202607)
        widget(app.button, "İstatistiksel testi çalıştır").click()
        app.run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        _, test = app.session_state["answers"]["statistical"]
        self.assertEqual(test["status"], "success")
        self.assertEqual(test["n"], 2052)

    def test_clarification_preserves_new_raw_question(self):
        query, rewrite, synthesis = MagicMock(), MagicMock(), MagicMock()
        rewrite.contextualize_query.side_effect = lambda question, history: question
        rewrite.route_question.return_value = "sql"
        rewrite.decompose_question.return_value = ("", ["Count"])
        query.execute_nl_query.return_value = build_result("Count", {"table": "demographics", "aggregates": [{"op": "count", "as": "n"}]}, "", [], ["n"], [(10,)])
        rewrite.assess_clarification_need.side_effect = [
            {"needs_clarification": False},
            {"needs_clarification": True, "options": [{"label": "Yaş", "context": "Yaş grupları"}]}]
        synthesis.synthesize_executive_summary.return_value = "Betimsel özet"
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-binding"}), patch("agent.get_hybrid_agent", return_value=(None,None,None,query,rewrite,synthesis)):
            app = self.app()
            widget(app.radio, "Çalışma modu").set_value("👤 Manuel Soru Modu")
            app.run(timeout=20)
            app.chat_input[0].set_value("İlk soru").run(timeout=20)
            app.chat_input[0].set_value("Yeni ve genel soru").run(timeout=20)
            widget(app.button, "Yaş").click().run(timeout=20)
            self.assertEqual(len(app.exception), 0)
            history = app.session_state["chat_history"]
            self.assertEqual(history[-1]["raw_question"], "Yeni ve genel soru")
            self.assertIn("Yaş grupları", history[-1]["answer"]["resolved_query"])
            self.assertEqual(history[-1]["answer"]["context"][0]["table"], "demographics")

    def test_reload_clears_previous_source_results_and_history(self):
        app = self.explorer(self.app())
        app.session_state["answers"] = {"old_source":"stale result"}
        app.session_state["chat_history"] = [{"user":"old source question"}]
        app.session_state["pending_clarification"] = {"raw":"old source question"}
        widget(app.button,"Bağlantıları yeniden yükle").click().run(timeout=20)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.session_state["answers"],{})
        self.assertEqual(app.session_state["chat_history"],[])
        self.assertIsNone(app.session_state["pending_clarification"])
