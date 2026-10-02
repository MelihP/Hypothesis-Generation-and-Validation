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
        json_content = """```json
{
  "H0": "Consideration düşüşü genel pazar trendidir.",
  "H1": "Consideration çöküşü tepe noktasının daraldığını gösterir.",
  "H2": "Tüketiciler doğrudan satın almaya geçmektedir.",
  "test_questions": [
    "2025 ve 2026 yıllarında consumer_journey aşamalarının tweet sayıları nedir?",
    "2026 yılında en çok bahsedilen topic_categories nedir?"
  ]
}
```"""
        mock_llm = RunnableLambda(lambda x: AIMessage(content=json_content))
        agent = RewriteNLAgent(llm=mock_llm)

        raw_text, sub_questions = agent.formulate_hypothesis("Genç kitle ve teknoloji", "Tablo şeması")
        self.assertIn("H0:", raw_text)
        self.assertIn("H1:", raw_text)
        self.assertIn("H2:", raw_text)
        self.assertEqual(len(sub_questions), 2)

    def test_formulate_competing_hypotheses(self):
        json_content = """```json
{
  "H0": "Consideration düşüşü genel pazar trendidir.",
  "H1": "Consideration çöküşü tepe noktasının daraldığını gösterir.",
  "H2": "Tüketiciler doğrudan satın almaya geçmektedir.",
  "test_questions": [
    "2025 ve 2026 yıllarında consumer_journey aşamalarının tweet sayıları nedir?",
    "2026 yılında en çok bahsedilen topic_categories nedir?"
  ]
}
```"""
        mock_llm = RunnableLambda(lambda x: AIMessage(content=json_content))
        agent = RewriteNLAgent(llm=mock_llm)

        hyp_dict, sub_questions = agent.formulate_competing_hypotheses("Genç kitle ve teknoloji", "Tablo şeması")
        self.assertEqual(hyp_dict["H0"], "Consideration düşüşü genel pazar trendidir.")
        self.assertEqual(hyp_dict["H1"], "Consideration çöküşü tepe noktasının daraldığını gösterir.")
        self.assertEqual(hyp_dict["H2"], "Tüketiciler doğrudan satın almaya geçmektedir.")
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
