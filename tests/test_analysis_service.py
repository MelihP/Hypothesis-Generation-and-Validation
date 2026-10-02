import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from agents.analysis_service import AnalysisService, DocumentSearch
from agents.results import build_result


def success():
    return build_result("Count", {"table": "demographics", "aggregates": [{"op": "count", "as": "users"}]}, "SELECT", [], ["users"], [(10,)])


class TestAnalysisService(unittest.TestCase):
    def setUp(self):
        self.query = MagicMock()
        self.rewrite = MagicMock()
        self.synthesis = MagicMock()
        self.rewrite.decompose_question.return_value = ("", ["First", "Second"])
        self.synthesis.synthesize_executive_summary.return_value = "Summary"
        self.service = AnalysisService(self.query, self.rewrite, self.synthesis)

    def test_all_failed_blocks_synthesis(self):
        self.query.execute_nl_query.side_effect = ValueError("Bad plan")
        answer = self.service.analyze("Question", source="sql")
        self.assertEqual(answer["status"], "failed")
        self.synthesis.synthesize_executive_summary.assert_not_called()

    def test_empty_and_truncated_block_synthesis(self):
        for kind in ("empty", "truncated"):
            result = success()
            result["status"] = "empty" if kind == "empty" else "success"
            result["truncated"] = kind == "truncated"
            self.query.execute_nl_query.side_effect = None
            self.query.execute_nl_query.return_value = result
            answer = self.service.analyze("Question", source="sql")
            self.assertEqual(answer["status"], "failed")
        self.synthesis.synthesize_executive_summary.assert_not_called()

    def test_partial_evidence_is_explicit(self):
        self.query.execute_nl_query.side_effect = [success(), ValueError("Bad")]
        answer = self.service.analyze("Question", source="sql")
        self.assertEqual(answer["status"], "partial")
        evidence = self.synthesis.synthesize_executive_summary.call_args.args[1]
        self.assertIn("KISMİ KANIT", evidence)
        self.assertNotIn("Bad", evidence)

    def test_hypothesis_requires_all_evidence(self):
        self.rewrite.formulate_competing_hypotheses.return_value = ({"H0": "A", "H1": "B"}, ["A", "B"])
        self.query.execute_nl_query.side_effect = [success(), ValueError("Bad")]
        answer = self.service.analyze("Question", mode="hypothesis", source="sql")
        self.assertEqual(answer["status"], "failed")
        self.synthesis.evaluate_competing_hypotheses.assert_not_called()

    def test_documents_and_hybrid_are_actually_called(self):
        retriever = MagicMock()
        retriever.invoke.return_value = [Document(page_content="Policy evidence", metadata={"source": "policy.pdf", "page": 2})]
        self.service.documents = DocumentSearch(retriever)
        self.synthesis.synthesize_executive_summary.return_value = "Policy summary [D1]"
        self.query.execute_nl_query.return_value = success()
        for route in ("documents", "hybrid"):
            self.rewrite.route_question.return_value = route
            answer = self.service.analyze("Question", source="auto")
            self.assertEqual(answer["status"], "success")
            self.assertEqual(answer["results"][-1]["rows"][0][:3], ["D1", "policy.pdf", 2])
            self.assertIn("policy.pdf", self.synthesis.synthesize_executive_summary.call_args.args[1])
        self.assertEqual(retriever.invoke.call_count, 2)

    def test_failed_document_search_is_not_silently_ignored(self):
        self.service.documents = MagicMock()
        self.service.documents.search.side_effect = ValueError("RAG unavailable")
        answer = self.service.analyze("Question", source="documents")
        self.assertEqual(answer["status"], "failed")
        self.synthesis.synthesize_executive_summary.assert_not_called()

    def test_invented_citations_are_rejected(self):
        self.service.documents = MagicMock()
        self.service.documents.search.return_value = {**success(), "source_type": "document", "rows": [["D1", "policy.pdf", 2, "", "Evidence"]]}
        self.synthesis.synthesize_executive_summary.return_value = "Claim [D99]"
        answer = self.service.analyze("Question", source="documents")
        self.assertEqual(answer["status"], "failed")
        self.assertEqual(answer["insight"], "")

    def test_single_period_blocks_predictive_synthesis(self):
        self.query.periods.return_value = [202607]
        with patch("agents.analysis_service.dataset_catalog", return_value={"consumer_journey": {"periods": [202607]}}):
            answer = self.service.analyze("Trend", mode="predictive", source="sql")
        self.assertEqual(answer["status"], "failed")
        self.synthesis.synthesize_predictive_insight.assert_not_called()
        self.rewrite.decompose_predictive_trends.assert_not_called()

    def test_autonomous_auto_uses_schema_database_without_document_routing(self):
        self.rewrite.route_question.return_value = "hybrid"
        self.query.execute_nl_query.return_value = success()
        self.service.documents = MagicMock()
        answer = self.service.analyze("Stage counts",mode="autonomous",source="auto")
        self.assertEqual(answer["status"],"success")
        self.rewrite.route_question.assert_not_called()
        self.service.documents.search.assert_not_called()
