"""Regression checks against the bundled dataset, without credentials or writes."""
import json
from pathlib import Path
import os
import unittest
from unittest.mock import patch

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.query_agent import QueryAgent
from agents.rewrite_nl_agent import RewriteNLAgent
from agents.credentials import configure_openai_credentials


class TestSchemaWorkflow(unittest.TestCase):
    def test_cloud_binding_is_used_without_overriding_existing_key(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-binding"}, clear=True):
            configure_openai_credentials()
            self.assertEqual(os.environ["OPENAI_API_KEY"], "test-binding")
        with patch.dict(os.environ, {"LLM_API_KEY": "test-binding", "OPENAI_API_KEY": "existing"}, clear=True):
            configure_openai_credentials()
            self.assertEqual(os.environ["OPENAI_API_KEY"], "existing")
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "insight_generation_bot.db"
        cls.db = SQLDatabase.from_uri(f"sqlite:///file:{path}?mode=ro&uri=true")

    def agent_for(self, plan):
        return QueryAgent(db=self.db, llm=RunnableLambda(lambda x: AIMessage(content=json.dumps(plan))))

    def test_literal_json_and_live_schema_reach_model(self):
        def respond(prompt):
            text = prompt.to_messages()[0].content
            self.assertIn('"joins": [', text)
            self.assertIn("CREATE TABLE demographics", text)
            self.assertNotIn("{schema}", text)
            self.assertIn("consumer_journey.author_id = demographics.user_id", text)
            return AIMessage(content='{"table":"demographics","limit":1}')
        QueryAgent(db=self.db, llm=RunnableLambda(respond)).generate_query_json("Bir kullanıcı getir")

    def test_demographic_journey_join_matches_direct_sql(self):
        plan = {
            "table": "consumer_journey",
            "joins": [{"table": "demographics", "on": {
                "left": "consumer_journey.author_id", "right": "demographics.user_id"}}],
            "group_by": ["demographics.age_group"],
            "aggregates": [{"op": "count_distinct", "column": "consumer_journey.tweet_id", "as": "tweets"}],
            "order_by": [{"column": "demographics.age_group"}]
        }
        result = self.agent_for(plan).execute_nl_query("Yaşa göre yolculuk tweet sayısı")
        expected = self.db.run("SELECT d.age_group, count(DISTINCT c.tweet_id) FROM consumer_journey c "
                               "JOIN demographics d ON c.author_id=d.user_id GROUP BY d.age_group ORDER BY d.age_group")
        self.assertEqual(result["result"], expected)

    def test_unknown_references_are_rejected_before_execution(self):
        for plan in [
            {"table": "twitter_tweets"},
            {"table": "demographics", "columns": ["age_range"]},
            {"table": "trending_topics", "group_by": ["topic_categories"]},
            {"table": "demographics", "filters": [{"column": "is_org", "op": "EQ", "value": 0}]},
        ]:
            with self.subTest(plan=plan), patch.object(self.db, "run") as run:
                with self.assertRaisesRegex(ValueError, "Şemada bulunmayan"):
                    self.agent_for(plan).execute_nl_query("Analiz")
                run.assert_not_called()

    def test_nested_unknown_column_is_rejected(self):
        plan = {"table": "consumer_journey", "filters": [{"column": "author_id", "op": "IN",
                "value": {"table": "demographics", "columns": ["id"]}}]}
        with self.assertRaisesRegex(ValueError, "Şemada bulunmayan"):
            self.agent_for(plan).execute_nl_query("Analiz")

    def test_aliases_and_aggregate_order_are_supported(self):
        plan = {"table": "demographics", "alias": "d", "group_by": ["d.age_group"],
                "aggregates": [{"op": "count", "column": "d.user_id", "as": "users"}],
                "order_by": [{"column": "users", "dir": "desc"}], "limit": 1}
        self.assertTrue(self.agent_for(plan).execute_nl_query("En kalabalık yaş grubu")["result"])

    def test_unsupported_model_output_is_clear(self):
        with self.assertRaisesRegex(ValueError, "Desteklenmeyen analiz"):
            self.agent_for({"unsupported": "Ürün alanı bulunmuyor"}).execute_nl_query("Ürünler")

    def test_rewrite_receives_shared_relationships(self):
        def respond(prompt):
            text = prompt.to_string()
            self.assertIn("consumer_journey.author_id = demographics.user_id", text)
            self.assertNotIn("twitter_tweets", text)
            return AIMessage(content="- Yaşa göre yolculuk dağılımı nedir?\n- Dönem kapsamı nedir?")
        agent = RewriteNLAgent(llm=RunnableLambda(respond))
        _, questions = agent.decompose_question("Yaş analizi", self.db.get_table_info())
        self.assertEqual(len(questions), 2)
