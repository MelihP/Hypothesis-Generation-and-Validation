import copy
from pathlib import Path
import sqlite3
import tempfile
import unittest

from langchain_community.utilities import SQLDatabase
from agents.query_agent import QueryAgent, compile_json_to_sql
from agents.sql_safety import read_query, validate_plan


class TestSafeExecution(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/"data.db"
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE items (id INTEGER, name TEXT, amount REAL)")
            db.executemany("INSERT INTO items VALUES (?, ?, ?)", [(1, "O'Reilly", 2), (2, "x", None), (3, "y", 0)])
        self.query = QueryAgent(db=SQLDatabase.from_uri(f"sqlite:///{self.path}"))

    def tearDown(self):
        self.temp.cleanup()

    def test_bound_values_do_not_change_sql(self):
        attack = "x' OR 1=1 --"
        result = self.query.execute_plan({"table": "items", "filters": [{"column": "name", "op": "EQ", "value": attack}]})
        self.assertNotIn(attack, result["sql"])
        self.assertIn(attack, result["parameters"])
        self.assertEqual(result["rows"], [])
        normal = self.query.execute_plan({"table": "items", "filters": [{"column": "name", "op": "EQ", "value": "O'Reilly"}]})
        self.assertEqual(normal["rows"][0][0], 1)

    def test_bad_plans_are_rejected(self):
        for extras in [
            {"limit": -1}, {"limit": 1001}, {"limit": "1"},
            {"filters": [{"column": "name", "op": "INVALID", "value": "x"}]},
            {"filters": [{"column": "name", "op": "BETWEEN", "value": [1]}]},
            {"joins": [{"table": "items", "on": "1 UNION SELECT id FROM items --"}]},
            {"unknown": "SQL"},
        ]:
            with self.subTest(extras=extras), self.assertRaises(ValueError):
                compile_json_to_sql({"table": "items", **extras})

    def test_read_only_and_authorizer(self):
        for sql in ["DELETE FROM items", "DROP TABLE items", "PRAGMA writable_schema=ON", "ATTACH DATABASE ':memory:' AS extra"]:
            with self.subTest(sql=sql), self.assertRaises(sqlite3.DatabaseError):
                read_query(self.path, sql)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM items").fetchone()[0], 3)

    def test_null_empty_list_and_bound_limit(self):
        nulls = self.query.execute_plan({"table": "items", "filters": [{"column": "amount", "op": "EQ", "value": None}]})
        self.assertEqual(nulls["rows"][0][0], 2)
        empty = self.query.execute_plan({"table": "items", "filters": [{"column": "id", "op": "IN", "value": []}]})
        self.assertEqual(empty["status"], "empty")
        result = self.query.execute_plan({"table": "items", "limit": 1})
        self.assertEqual(len(result["rows"]), 1)
        self.assertTrue(result["truncated"])

    def test_timeout_and_depth_limits(self):
        with self.assertRaises(TimeoutError):
            read_query(self.path, "WITH RECURSIVE numbers(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM numbers) SELECT sum(x) FROM numbers", timeout=.001)
        plan = {"table": "items", "columns": ["id"]}
        for _ in range(5):
            plan = {"table": "items", "columns": ["id"], "filters": [{"column": "id", "op": "IN", "value": copy.deepcopy(plan)}]}
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_composite_join_conditions(self):
        plan = {"table": "items", "alias": "a", "columns": ["a.id"], "joins": [{"table": "items", "alias": "b",
                "on": [{"left": "a.id", "right": "b.id"}, {"left": "a.name", "right": "b.name"}]}]}
        self.assertEqual(len(self.query.execute_plan(plan)["rows"]), 3)

    def test_model_singleton_sort_and_optional_null_are_normalized(self):
        import json
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        plan = {"table":"items", "columns":["id"], "filters":None,
                "order_by":{"column":"id","dir":"desc"},"limit":3}
        self.query._llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(plan)))
        result = self.query.execute_nl_query("IDs descending")
        self.assertEqual(result["rows"], [[3],[2],[1]])
        self.assertNotIn("filters",result["json_query"])
        self.assertEqual(result["json_query"]["order_by"],[{"column":"id","dir":"desc"}])
        plan["order_by"] = "id DESC; DELETE FROM items"
        with self.assertRaises(ValueError):
            self.query.execute_nl_query("IDs descending")
