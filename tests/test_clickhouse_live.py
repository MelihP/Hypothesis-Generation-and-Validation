"""Opt-in integration against a disposable local ClickHouse, never production."""
import os
import unittest
import clickhouse_connect

from agents.database import ClickHouseBackend
from agents.query_agent import QueryAgent
from agents.explorer import independent_comparison, period_filters


@unittest.skipUnless(os.environ.get("LOCAL_CLICKHOUSE_TEST") == "1", "Local ClickHouse integration is opt-in")
class TestLocalClickHouse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = clickhouse_connect.get_client(host="127.0.0.1", port=18123,
                            username="hypothesis_test", password="local-test-only")
        cls.client.command("CREATE DATABASE IF NOT EXISTS hypothesis_fixture")
        cls.client.command("CREATE TABLE IF NOT EXISTS hypothesis_fixture.users (id UInt64, gender String, age_group String) ENGINE=Memory")
        cls.client.command("CREATE TABLE IF NOT EXISTS hypothesis_fixture.tweet_predictions (tweet_id UInt64, author_id UInt64, task_name String, category_value Array(String), created_at DateTime) ENGINE=Memory")
        # This database is test-only and this class owns these tables.
        cls.client.command("TRUNCATE TABLE hypothesis_fixture.users")
        cls.client.command("TRUNCATE TABLE hypothesis_fixture.tweet_predictions")
        cls.client.command("INSERT INTO hypothesis_fixture.users SELECT number+1, if(number<20,'female','male'), if(number<20,'18-29','30-39') FROM numbers(42)")
        cls.client.command("INSERT INTO hypothesis_fixture.tweet_predictions SELECT number+1000, number+1, 'emotion', [if(number<15 OR (number>=20 AND number<25),'positive','negative')], if(number<20,toDateTime('2026-07-01'),toDateTime('2026-08-01')) FROM numbers(40)")
        cls.client.command("INSERT INTO hypothesis_fixture.tweet_predictions VALUES (2000,41,'emotion',['positive'],'2026-07-01'),(2001,41,'emotion',['negative'],'2026-07-01'),(2002,42,'emotion',['positive','negative'],'2026-07-01')")
        native = clickhouse_connect.get_client(host="127.0.0.1", port=18123, username="hypothesis_test", password="local-test-only", database="hypothesis_fixture")
        cls.backend = ClickHouseBackend(native, "hypothesis_fixture")
        cls.query = QueryAgent(db=cls.backend, dialect="clickhouse")

    @classmethod
    def tearDownClass(cls):
        cls.backend.client.close()
        cls.client.command("DROP DATABASE hypothesis_fixture")
        cls.client.close()

    def test_native_catalog_and_periods(self):
        self.assertEqual(self.query.catalog()["tweet_predictions"]["types"]["category_value"], "Array(String)")
        self.assertEqual(self.query.periods("tweet_predictions"), [202607,202608])

    def test_bound_array_filters_and_injection(self):
        plan = {"table":"tweet_predictions","filters":[{"column":"category_value","op":"HAS_ANY","value":["positive"]}],
                "aggregates":[{"op":"count_distinct","column":"author_id","as":"users"}]}
        result = self.query.execute_plan(plan)
        self.assertEqual(result["rows"], [[22]])
        attack = "emotion' OR 1=1 --\\"
        result = self.query.execute_plan({"table":"tweet_predictions","filters":[{"column":"task_name","op":"EQ","value":attack}]})
        self.assertEqual(result["rows"], [])
        self.assertNotIn(attack,result["sql"])

    def test_join_then_array_join_and_alias_validation(self):
        plan = {"table":"tweet_predictions","alias":"p","joins":[{"table":"users","alias":"u","on":{"left":"p.author_id","right":"u.id"}}],
                "array_joins":[{"column":"p.category_value","as":"tag"}],"group_by":["u.gender","tag"],
                "aggregates":[{"op":"count_distinct","column":"p.author_id","as":"users"}]}
        result = self.query.execute_plan(plan)
        self.assertEqual(len(result["rows"]),4)
        self.assertTrue(result["warnings"])
        bad = {"table":"users","array_joins":[{"column":"gender","as":"tag"}],"columns":["tag"]}
        with self.assertRaises(ValueError):
            self.query.execute_plan(bad)

    def test_readonly_and_clip(self):
        result = self.query.execute_plan({"table":"users","limit":2})
        self.assertEqual(len(result["rows"]),2)
        self.assertTrue(result["truncated"])
        with self.assertRaises(ValueError):
            self.backend.read("DELETE FROM users WHERE id=1")
        with self.assertRaises(Exception):
            self.backend.read("SELECT sleep(2)",timeout=1)

    def test_independent_statistics_exclude_duplicates_and_multi_labels(self):
        cross,test = independent_comparison(self.query,"tweet_predictions","users","author_id","id","gender","category_value",
                                            [{"column":"task_name","op":"EQ","value":"emotion"}])
        self.assertEqual(int(cross.values.sum()),40)
        self.assertEqual(test["scope"]["excluded_users"],2)
        self.assertEqual(test["status"],"success")
        self.assertLess(test["p_value"],.01)

    def test_date_scope_filters(self):
        filters = period_filters(self.query.catalog()["tweet_predictions"],202608)
        result = self.query.execute_plan({"table":"tweet_predictions","filters":filters,"aggregates":[{"op":"count","as":"n"}]})
        self.assertEqual(result["rows"],[[20]])

    def test_streamlit_array_builder_uses_native_clickhouse(self):
        from unittest.mock import patch
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from tests.test_app_workflows import APP, widget
        st.cache_resource.clear()
        st.cache_data.clear()
        try:
            with patch('agent.get_database_connection', return_value=(self.backend, 'clickhouse://local-test', 'clickhouse')):
                app = AppTest.from_file(APP).run(timeout=20)
                widget(app.radio, 'Çalışma modu').set_value('📊 Veri Sorgulama ve İstatistik')
                app.run(timeout=20)
                widget(app.selectbox, 'Tablo').set_value('tweet_predictions')
                app.run(timeout=20)
                widget(app.multiselect, 'Gruplama sütunları (çapraz tablo için iki sütun)').set_value(['category_value'])
                widget(app.selectbox, 'Metrik').set_value('Farklı tweet sayısı')
                app.run(timeout=20)
                widget(app.checkbox, 'Dizi etiketlerini ayrı satırlara aç').check()
                widget(app.button, 'Tabloyu oluştur').click()
                app.run(timeout=20)
                self.assertEqual(len(app.exception), 0)
                result = app.session_state['answers']['builder']
                self.assertEqual(result['status'], 'success')
                self.assertEqual(sum(row[1] for row in result['rows']), 44)
                self.assertIn('ARRAY JOIN', result['sql'])
                self.assertTrue(result['warnings'])
        finally:
            st.cache_resource.clear()
            st.cache_data.clear()
