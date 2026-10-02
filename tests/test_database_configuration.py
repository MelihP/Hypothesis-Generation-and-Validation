"""Existing deployment secrets remain compatible; failures never switch data silently."""
import unittest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager
import agent


class TestDatabaseConfiguration(unittest.TestCase):
    @contextmanager
    def configure(self, values):
        getter = lambda key, default="": values.get(key, default)
        with patch("agent.get_secret", side_effect=getter), patch("agents.config.get_secret", side_effect=getter):
            yield

    def test_existing_tls_settings_and_credentials_pass_to_native_driver(self):
        config = {'CLICKHOUSE_HOST':'fixture.example', 'CLICKHOUSE_PORT':'443',
                  'CLICKHOUSE_USERNAME':'readonly', 'CLICKHOUSE_PASSWORD':'password-with:@/',
                  'CLICKHOUSE_DB':'analytics', 'CLICKHOUSE_VERIFY':'false'}
        with self.configure(config), patch('clickhouse_connect.get_client', return_value=MagicMock()) as client, patch('agents.database.ClickHouseBackend.catalog', return_value={'users':{}}):
            backend, uri, dialect = agent.get_database_connection()
        self.assertTrue(client.call_args.kwargs['secure'])
        self.assertFalse(client.call_args.kwargs['verify'])
        self.assertEqual(client.call_args.kwargs['password'],config['CLICKHOUSE_PASSWORD'])
        self.assertTrue(backend.tls_verification_disabled)
        self.assertNotIn(config['CLICKHOUSE_PASSWORD'],uri)
        self.assertEqual(dialect,'clickhouse')

    def test_connection_error_does_not_fallback_or_expose_exception(self):
        with self.configure({'CLICKHOUSE_HOST':'fixture.example'}), patch('clickhouse_connect.get_client', side_effect=RuntimeError('secret-password')), patch('agent.SQLiteBackend') as sqlite:
            with self.assertRaises(ValueError) as error:
                agent.get_database_connection()
        sqlite.assert_not_called()
        self.assertNotIn('secret-password',str(error.exception))

    def test_explicit_fallback_is_marked(self):
        with self.configure({'CLICKHOUSE_HOST':'fixture.example','ALLOW_SQLITE_FALLBACK':'true'}), patch('clickhouse_connect.get_client', side_effect=RuntimeError('connection')), patch('agent.SQLiteBackend') as sqlite:
            agent.get_database_connection()
        self.assertTrue(sqlite.call_args.kwargs['fallback'])
