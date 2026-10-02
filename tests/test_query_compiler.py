import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.sql_compiler import (
    compile_json_to_sql,
    quote_ident,
    _lit,
    _contains_lit,
    _format_array_lit,
    _compile_join,
)
from agents.query_agent import QueryAgent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class TestQueryCompiler(unittest.TestCase):

    # --- 1. IDENTIFIER QUOTING & DIALECTS ---

    def test_quote_ident_sqlite(self):
        self.assertEqual(quote_ident("demographics", dialect="sqlite"), '"demographics"')
        self.assertEqual(quote_ident('col"name', dialect="sqlite"), '"col""name"')
        self.assertEqual(quote_ident("t.user_id", dialect="sqlite"), '"t"."user_id"')
        self.assertEqual(quote_ident("json_each(tweets.topics)", dialect="sqlite"), 'json_each("tweets"."topics")')

    def test_quote_ident_clickhouse(self):
        self.assertEqual(quote_ident("tweet_predictions", dialect="clickhouse"), '`tweet_predictions`')
        self.assertEqual(quote_ident('col`name', dialect="clickhouse"), '`col``name`')
        self.assertEqual(quote_ident("t.author_id", dialect="clickhouse"), '`t`.`author_id`')

    def test_quote_ident_dangerous(self):
        with self.assertRaises(ValueError):
            quote_ident("users; DROP TABLE users;--")
        with self.assertRaises(ValueError):
            quote_ident("tweets/*comment*/")

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

    # --- 2. SQLITE DIALECT COMPILATION ---

    def test_simple_select_sqlite(self):
        spec = {
            "table": "demographics",
            "columns": ["user_id", "age_group"],
            "limit": 10
        }
        sql = compile_json_to_sql(spec, dialect="sqlite")
        self.assertEqual(sql, 'SELECT "user_id", "age_group" FROM "demographics" LIMIT 10')

    def test_aggregation_and_group_by_sqlite(self):
        spec = {
            "table": "demographics",
            "aggregates": [
                {"op": "count", "column": "user_id", "as": "total_users"}
            ],
            "group_by": ["age_group"],
            "order_by": [{"column": "total_users", "dir": "desc"}],
            "limit": 5
        }
        sql = compile_json_to_sql(spec, dialect="sqlite")
        self.assertEqual(sql, 'SELECT "age_group", count("user_id") AS "total_users" FROM "demographics" GROUP BY "age_group" ORDER BY "total_users" DESC LIMIT 5')

    def test_filters_equality_and_gt_sqlite(self):
        spec = {
            "table": "demographics",
            "columns": ["user_id"],
            "filters": [
                {"column": "age_group", "op": "EQ", "value": "18-29"},
                {"column": "user_id", "op": "GT", "value": 1000}
            ]
        }
        sql = compile_json_to_sql(spec, dialect="sqlite")
        self.assertEqual(sql, "SELECT \"user_id\" FROM \"demographics\" WHERE \"age_group\" = '18-29' AND \"user_id\" > 1000")

    def test_filters_in_and_between_sqlite(self):
        spec = {
            "table": "demographics",
            "columns": ["user_id"],
            "filters": [
                {"column": "age_group", "op": "IN", "value": ["18-29", "30-39"]},
                {"column": "user_id", "op": "BETWEEN", "value": [100, 500]}
            ]
        }
        sql = compile_json_to_sql(spec, dialect="sqlite")
        self.assertEqual(sql, "SELECT \"user_id\" FROM \"demographics\" WHERE \"age_group\" IN ('18-29', '30-39') AND \"user_id\" BETWEEN 100 AND 500")

    def test_joins_compilation_sqlite(self):
        spec = {
            "table": "twitter_tweets",
            "joins": [
                {
                    "type": "INNER",
                    "table": "demo_brand_users",
                    "on": {"left": "twitter_tweets.author_id", "right": "demo_brand_users.id"}
                }
            ],
            "columns": ["twitter_tweets.id", "demo_brand_users.age_range"],
            "limit": 10
        }
        sql = compile_json_to_sql(spec, dialect="sqlite")
        expected = 'SELECT "twitter_tweets"."id", "demo_brand_users"."age_range" FROM "twitter_tweets" INNER JOIN "demo_brand_users" ON "twitter_tweets"."author_id" = "demo_brand_users"."id" LIMIT 10'
        self.assertEqual(sql, expected)

    def test_nested_subquery_in_filter_sqlite(self):
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
        sql = compile_json_to_sql(spec, dialect="sqlite")
        expected = "SELECT \"tweet_id\", count(\"tweet_id\") AS \"complaint_count\" FROM \"consumer_journey\" WHERE \"journey_stage\" = 'Complaint' AND \"author_id\" IN (SELECT \"user_id\" FROM \"demographics\" WHERE \"age_group\" = '18-29') GROUP BY \"tweet_id\" ORDER BY \"complaint_count\" DESC LIMIT 50"
        self.assertEqual(sql, expected)

    # --- 3. CLICKHOUSE DIALECT COMPILATION ---

    def test_simple_select_clickhouse(self):
        spec = {
            "table": "tweet_predictions",
            "columns": ["tweet_id", "task_name"],
            "limit": 20
        }
        sql = compile_json_to_sql(spec, dialect="clickhouse")
        self.assertEqual(sql, "SELECT `tweet_id`, `task_name` FROM `tweet_predictions` LIMIT 20")

    def test_clickhouse_array_has_filter(self):
        spec = {
            "table": "tweet_predictions",
            "columns": ["tweet_id"],
            "filters": [
                {"column": "task_name", "op": "EQ", "value": "emotion"},
                {"column": "category_value", "op": "HAS", "value": "mutluluk"}
            ],
            "limit": 10
        }
        sql = compile_json_to_sql(spec, dialect="clickhouse")
        self.assertEqual(sql, "SELECT `tweet_id` FROM `tweet_predictions` WHERE `task_name` = 'emotion' AND has(`category_value`, 'mutluluk') LIMIT 10")

    def test_clickhouse_array_has_any_filter(self):
        spec = {
            "table": "tweet_predictions",
            "columns": ["tweet_id"],
            "filters": [
                {"column": "category_value", "op": "HAS_ANY", "value": ["mutluluk", "sevgi"]}
            ]
        }
        sql = compile_json_to_sql(spec, dialect="clickhouse")
        self.assertEqual(sql, "SELECT `tweet_id` FROM `tweet_predictions` WHERE hasAny(`category_value`, ['mutluluk', 'sevgi'])")

    def test_clickhouse_array_join(self):
        spec = {
            "table": "tweet_predictions",
            "columns": ["tag"],
            "aggregates": [{"op": "count", "column": "tweet_id", "as": "cnt"}],
            "array_joins": [{"column": "category_value", "as": "tag"}],
            "group_by": ["tag"],
            "order_by": [{"column": "cnt", "dir": "desc"}],
            "limit": 10
        }
        sql = compile_json_to_sql(spec, dialect="clickhouse")
        self.assertEqual(sql, "SELECT `tag`, count(`tweet_id`) AS `cnt` FROM `tweet_predictions` ARRAY JOIN `category_value` AS `tag` GROUP BY `tag` ORDER BY `cnt` DESC LIMIT 10")

    def test_clickhouse_nested_subquery(self):
        spec = {
            "table": "tweet_predictions",
            "columns": ["tweet_id"],
            "filters": [
                {"column": "task_name", "op": "EQ", "value": "emotion"},
                {
                    "column": "author_id",
                    "op": "IN",
                    "value": {
                        "table": "users",
                        "columns": ["id"],
                        "filters": [{"column": "gender", "op": "EQ", "value": "female"}]
                    }
                }
            ],
            "limit": 5
        }
        sql = compile_json_to_sql(spec, dialect="clickhouse")
        expected = "SELECT `tweet_id` FROM `tweet_predictions` WHERE `task_name` = 'emotion' AND `author_id` IN (SELECT `id` FROM `users` WHERE `gender` = 'female') LIMIT 5"
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

        agent = QueryAgent(llm=mock_llm, db=mock_db, schema="mock_schema", dialect="sqlite")
        out = agent.execute_nl_query("Yaş dağılımı nedir?")
        
        self.assertEqual(out["question"], "Yaş dağılımı nedir?")
        self.assertEqual(out["result"], "[('18-29', 5420)]")
        self.assertIn('SELECT "age_group", count("user_id") AS "cnt" FROM "demographics"', out["sql"])

    def test_query_agent_compilation_failure_logged(self):
        mock_db = MagicMock()
        # Invalid JSON missing "table" field
        json_response = """```json
{
  "columns": ["age_group"]
}
```"""
        mock_llm = RunnableLambda(lambda x: AIMessage(content=json_response))
        agent = QueryAgent(llm=mock_llm, db=mock_db, schema="mock_schema", dialect="sqlite")

        with self.assertRaises(ValueError) as ctx:
            agent.execute_nl_query("Hatalı sorgu")
        self.assertIn("table", str(ctx.exception).lower())

    def test_query_agent_db_execution_failure_logged(self):
        mock_db = MagicMock()
        mock_db.run.side_effect = RuntimeError("no such column: fake_col")

        json_response = """```json
{
  "table": "demographics",
  "columns": ["fake_col"]
}
```"""
        mock_llm = RunnableLambda(lambda x: AIMessage(content=json_response))
        agent = QueryAgent(llm=mock_llm, db=mock_db, schema="mock_schema", dialect="sqlite")

        with self.assertRaises(RuntimeError) as ctx:
            agent.execute_nl_query("Varolmayan sütun sorgusu")
        self.assertIn("fake_col", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
