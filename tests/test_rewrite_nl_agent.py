import unittest
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
        content = (
            "Hipotez (H1): Genç kitle teknoloji markalarına daha olumlu yaklaşmaktadır.\n"
            "- Demographics ve tweet_predictions tablolarını JOIN yaparak yaş gruplarını hesaplayın.\n"
            "- Pozitif sentiment oranlarını karşılaştırın."
        )
        mock_llm = RunnableLambda(lambda x: AIMessage(content=content))
        agent = RewriteNLAgent(llm=mock_llm)

        raw_text, sub_questions = agent.formulate_hypothesis("Genç kitle ve teknoloji", "Tablo şeması")
        self.assertIn("Hipotez (H1)", raw_text)
        self.assertEqual(len(sub_questions), 2)

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
