import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.prompts import (
    MARKETING_CONCEPT_DEFINITIONS,
    BUSINESS_HEURISTIC_RULES,
    get_domain_context_prompt,
    build_hypothesis_synthesis_prompt,
    QUERY_GENERATOR_SYSTEM_PROMPT,
    SQL_AGENT_PREFIX,
    CONTEXTUALIZE_QUERY_PROMPT,
    ASSESS_CLARIFICATION_NEED_PROMPT,
    MACRO_QUESTION_PROMPT,
    DECOMPOSE_QUESTION_PROMPT,
    COMPETING_HYPOTHESES_PROMPT,
    PREDICTIVE_TRENDS_PROMPT,
    EXECUTIVE_SUMMARY_PROMPT,
    COMPETING_HYPOTHESES_EVALUATION_PROMPT,
    PREDICTIVE_INSIGHT_PROMPT,
)


class TestPromptsIsolation(unittest.TestCase):

    def test_domain_prompts_exported_and_identical(self):
        self.assertTrue(len(MARKETING_CONCEPT_DEFINITIONS.strip()) > 0)
        self.assertTrue(len(BUSINESS_HEURISTIC_RULES.strip()) > 0)
        self.assertIn("CONSUMER JOURNEY", MARKETING_CONCEPT_DEFINITIONS)
        self.assertIn("STRATEJİK ÇIKARIM", BUSINESS_HEURISTIC_RULES)
        self.assertIn(MARKETING_CONCEPT_DEFINITIONS.strip(), get_domain_context_prompt())
        
        synth = build_hypothesis_synthesis_prompt("Test Hyp", "SELECT 1")
        self.assertIn("Test Hyp", synth)
        self.assertIn("SELECT 1", synth)

    def test_query_prompts_exported(self):
        self.assertIn("JSON Sorgu Planlama Ajanısın", QUERY_GENERATOR_SYSTEM_PROMPT)
        self.assertIn("{schema}", QUERY_GENERATOR_SYSTEM_PROMPT)
        self.assertIn("SQL Danışmanısın", SQL_AGENT_PREFIX)

    def test_rewrite_prompts_exported(self):
        self.assertIn("Konuşma Bağlamı", CONTEXTUALIZE_QUERY_PROMPT)
        self.assertIn("Niyet Belirleme", ASSESS_CLARIFICATION_NEED_PROMPT)
        self.assertIn("pazarlama direktörüsün", MACRO_QUESTION_PROMPT)
        self.assertIn("veri analistisin", DECOMPOSE_QUESTION_PROMPT)
        self.assertIn("H0", COMPETING_HYPOTHESES_PROMPT)
        self.assertIn("tahminleme veri bilimcisisin", PREDICTIVE_TRENDS_PROMPT)

    def test_synthesis_prompts_exported(self):
        self.assertIn("Pazarlama Direktörüsün (CMO)", EXECUTIVE_SUMMARY_PROMPT)
        self.assertIn("Baş Ekonometrist", COMPETING_HYPOTHESES_EVALUATION_PROMPT)
        self.assertIn("Büyüme Stratejistisin", PREDICTIVE_INSIGHT_PROMPT)


if __name__ == "__main__":
    unittest.main()
