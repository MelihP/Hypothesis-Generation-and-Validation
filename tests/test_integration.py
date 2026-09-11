import unittest
from unittest.mock import MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.query_agent import QueryAgent
from agents.rewrite_nl_agent import RewriteNLAgent
from logger import log_query
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class TestAgentsIntegration(unittest.TestCase):

    def test_full_pipeline_mock(self):
        # 1. Setup mock DB
        mock_db = MagicMock()
        mock_db.get_table_info.return_value = "CREATE TABLE demographics (user_id INTEGER, age_group TEXT);"
        mock_db.run.return_value = "[('18-29', 5420)]"

        # 2. Step A: Rewrite macro question to sub-questions
        rewrite_resp = AIMessage(content="- Yaş gruplarına göre kullanıcı sayıları nedir?")
        rewrite_llm = RunnableLambda(lambda x: rewrite_resp)
        rewrite_agent = RewriteNLAgent(llm=rewrite_llm)

        raw_text, sub_questions = rewrite_agent.decompose_question("Genel demografi analizi", mock_db.get_table_info())
        self.assertEqual(len(sub_questions), 1)
        sub_q = sub_questions[0]

        # 3. Step B: Query Agent converts sub-question to JSON and executes SQL
        query_json_text = """{
            "table": "demographics",
            "group_by": ["age_group"],
            "aggregates": [{"op": "count", "column": "user_id", "as": "cnt"}]
        }"""
        query_llm = RunnableLambda(lambda x: AIMessage(content=query_json_text))
        query_agent = QueryAgent(llm=query_llm, db=mock_db, schema=mock_db.get_table_info())

        exec_res = query_agent.execute_nl_query(sub_q)
        self.assertEqual(exec_res["result"], "[('18-29', 5420)]")
        self.assertIn('GROUP BY "age_group"', exec_res["sql"])

        # 4. Step C: Log the execution
        log_query(
            question=exec_res["question"],
            json_query=exec_res["json_query"],
            sql=exec_res["sql"],
            result=exec_res["result"],
            duration_ms=34.1
        )


if __name__ == "__main__":
    unittest.main()
