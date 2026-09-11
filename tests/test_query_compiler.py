import unittest
from unittest.mock import MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.query_agent import compile_json_to_sql, quote_ident, _lit, _contains_lit, QueryAgent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class TestQueryCompiler(unittest.TestCase):

    def test_quote_ident_safe(self):
        self.assertEqual(quote_ident("demographics"), '"demographics"')
        self.assertEqual(quote_ident('col"name'), '"col""name"')

    def test_quote_ident_dangerous(self):
        with self.assertRaises(ValueError):
            quote_ident("users; DROP TABLE users;")
        with self.assertRaises(ValueError):
            quote_ident("users--comment")
        with self.assertRaises(ValueError):
            quote_ident("users/*comment*/")

    def test_lit_formatting(self):
        self.assertEqual(_lit(None), "NULL")
        self.assertEqual(_lit(True), "1")
        self.assertEqual(_lit(False), "0")
        self.assertEqual(_lit(42), "42")
        self.assertEqual(_lit(3.14), "3.14")
        self.assertEqual(_lit("Trendyol"), "'Trendyol'")
        self.assertEqual(_lit("O'Reilly"), "'O''Reilly'")

    def test_contains_lit(self):
        self.assertEqual(_contains_lit("Nike"), "'%Nike%'")
        self.assertEqual(_contains_lit("O'Connor"), "'%O''Connor%'")

    def test_simple_select(self):
        spec = {
            "table": "demographics",
            "columns": ["user_id", "age_group"],
            "limit": 10
        }
        sql = compile_json_to_sql(spec)
        self.assertEqual(sql, 'SELECT "user_id", "age_group" FROM "demographics" LIMIT 10')

    def test_aggregation_and_group_by(self):
        spec = {
            "table": "demographics",
            "aggregates": [
                {"op": "count", "column": "user_id", "as": "total_users"}
            ],
            "group_by": ["age_group"],
            "order_by": [{"column": "total_users", "dir": "desc"}],
            "limit": 5
        }
        sql = compile_json_to_sql(spec)
        self.assertEqual(sql, 'SELECT "age_group", count("user_id") AS "total_users" FROM "demographics" GROUP BY "age_group" ORDER BY "total_users" DESC LIMIT 5')

    def test_filters_equality_and_gt(self):
        spec = {
            "table": "demographics",
            "columns": ["user_id"],
            "filters": [
                {"column": "age_group", "op": "EQ", "value": "18-29"},
                {"column": "user_id", "op": "GT", "value": 1000}
            ]
        }
        sql = compile_json_to_sql(spec)
        self.assertEqual(sql, "SELECT \"user_id\" FROM \"demographics\" WHERE \"age_group\" = '18-29' AND \"user_id\" > 1000")

    def test_filters_in_and_between(self):
        spec = {
            "table": "demographics",
            "columns": ["user_id"],
            "filters": [
                {"column": "age_group", "op": "IN", "value": ["18-29", "30-39"]},
                {"column": "user_id", "op": "BETWEEN", "value": [100, 500]}
            ]
        }
        sql = compile_json_to_sql(spec)
        self.assertEqual(sql, "SELECT \"user_id\" FROM \"demographics\" WHERE \"age_group\" IN ('18-29', '30-39') AND \"user_id\" BETWEEN 100 AND 500")

    def test_nested_subquery_in_filter(self):
        spec = {
            "table": "consumer_journey",
            "columns": ["tweet_id"],
            "aggregates": [{"op": "count", "column": "tweet_id", "as": "complaint_count"}],
            "filters": [
                {"column": "journey_stage", "op": "EQ", "value": "Complaint"},
                {
                    "column": "author_id",
                    "op": "IN",
                    "value": {
                        "table": "demographics",
                        "columns": ["user_id"],
                        "filters": [{"column": "age_group", "op": "EQ", "value": "18-29"}]
                    }
                }
            ],
            "group_by": ["tweet_id"],
            "order_by": [{"column": "complaint_count", "dir": "desc"}],
            "limit": 50
        }
        sql = compile_json_to_sql(spec)
        expected = "SELECT \"tweet_id\", count(\"tweet_id\") AS \"complaint_count\" FROM \"consumer_journey\" WHERE \"journey_stage\" = 'Complaint' AND \"author_id\" IN (SELECT \"user_id\" FROM \"demographics\" WHERE \"age_group\" = '18-29') GROUP BY \"tweet_id\" ORDER BY \"complaint_count\" DESC LIMIT 50"
        self.assertEqual(sql, expected)

    def test_missing_table_raises_error(self):
        spec = {"columns": ["user_id"]}
        with self.assertRaises(ValueError):
            compile_json_to_sql(spec)

    def test_query_agent_generate_json_and_execute(self):
        mock_db = MagicMock()
        mock_db.get_table_info.return_value = "CREATE TABLE demographics (user_id INTEGER, age_group TEXT);"
        mock_db.run.return_value = "[('18-29', 5420)]"

        json_response = """```json
{
  "table": "demographics",
  "group_by": ["age_group"],
  "aggregates": [{"op": "count", "column": "user_id", "as": "cnt"}],
  "order_by": [{"column": "cnt", "dir": "desc"}]
}
```"""
        mock_llm = RunnableLambda(lambda x: AIMessage(content=json_response))

        agent = QueryAgent(llm=mock_llm, db=mock_db, schema="mock_schema")
        out = agent.execute_nl_query("Yaş dağılımı nedir?")
        
        self.assertEqual(out["question"], "Yaş dağılımı nedir?")
        self.assertEqual(out["result"], "[('18-29', 5420)]")
        self.assertIn('SELECT "age_group", count("user_id") AS "cnt" FROM "demographics"', out["sql"])


if __name__ == "__main__":
    unittest.main()
