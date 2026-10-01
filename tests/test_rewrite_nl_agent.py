import unittest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.rewrite_nl_agent import RewriteNLAgent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class TestRewriteNLAgent(unittest.TestCase):

    def test_generate_macro_question(self):
        mock_resp = AIMessage(content="Hangi yaş grubu Trendyol hakkında daha pozitiftir?")
        mock_llm = RunnableLambda(lambda x: mock_resp)
        agent = RewriteNLAgent(llm=mock_llm)

        res = agent.generate_macro_question("Veritabanı özeti metni")
        self.assertEqual(res, "Hangi yaş grubu Trendyol hakkında daha pozitiftir?")

    def test_decompose_question(self):
        content = (
            "Giriş açıklaması\n"
            "- 1. Yaş gruplarına göre Trendyol duygu dağılımı nedir?\n"
            "- 2. Şirket müşteri vizyon belgelerinde ne belirtilmiş?\n"
        )
        mock_llm = RunnableLambda(lambda x: AIMessage(content=content))
        agent = RewriteNLAgent(llm=mock_llm)

        raw_text, sub_questions = agent.decompose_question("Makro soru", "Tablo şeması")
        self.assertEqual(len(sub_questions), 2)
        self.assertIn("Trendyol", sub_questions[0])
        self.assertIn("vizyon", sub_questions[1])

    def test_formulate_hypothesis(self):
        content = json.dumps({
            "H0": "Duygu dağılımı yaş gruplarında aynıdır.",
            "H1": "Genç kitlenin duygu dağılımı farklıdır.",
            "H2": "Fark örneklem bileşiminden kaynaklanmaktadır.",
            "test_questions": ["Duygu verisinin dönem kapsamı nedir?", "Yaşa göre duygu dağılımı nedir?"]
        })
        mock_llm = RunnableLambda(lambda x: AIMessage(content=content))
        agent = RewriteNLAgent(llm=mock_llm)

        raw_text, sub_questions = agent.formulate_hypothesis("Genç kitle ve teknoloji", "Tablo şeması")
        self.assertIn("H1: Genç kitlenin duygu dağılımı farklıdır.", raw_text)
        self.assertEqual(len(sub_questions), 2)

    def test_invalid_hypothesis_does_not_invent_unrelated_queries(self):
        agent = RewriteNLAgent(llm=RunnableLambda(lambda x: AIMessage(content="Geçersiz çıktı")))
        with self.assertRaisesRegex(ValueError, "Hipotez planı geçersiz"):
            agent.formulate_competing_hypotheses("Yaş ve duygu", "Tablo şeması")

    def test_decompose_predictive_trends(self):
        content = (
            "Açıklama\n"
            "- Tarih bazında aylık tweet sayılarının değişim trendi nedir?\n"
            "- En çok etkileşim alan kategorilerin sıralaması nasıldır?"
        )
        mock_llm = RunnableLambda(lambda x: AIMessage(content=content))
        agent = RewriteNLAgent(llm=mock_llm)

        raw_text, sub_questions = agent.decompose_predictive_trends("Gelecek trendi", "Tablo şeması")
        self.assertEqual(len(sub_questions), 2)


if __name__ == "__main__":
    unittest.main()
