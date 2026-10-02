"""Deployment configuration and lazy Pinecone wiring without real credentials."""
import os
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from agents.config import get_secret
from agents.credentials import configure_openai_credentials
from agents.analysis_service import DocumentSearch


class TestStreamlitSecrets(unittest.TestCase):
    def test_flat_nested_and_environment_precedence(self):
        secrets = {'PINECONE_API_KEY':'flat-key', 'pinecone':{'api_key':'nested-key'},
                   'clickhouse':{'host':'fixture', 'secure':True, 'port':8443}}
        with patch('agents.config.st.secrets',secrets), patch.dict(os.environ, {'PINECONE_API_KEY':'env-key'}):
            self.assertEqual(get_secret('PINECONE_API_KEY'),'flat-key')
            self.assertEqual(get_secret('CLICKHOUSE_HOST'),'fixture')
            self.assertEqual(get_secret('CLICKHOUSE_PORT'),'8443')
            self.assertEqual(get_secret('CLICKHOUSE_SECURE'),'True')
            secrets.pop('PINECONE_API_KEY')
            self.assertEqual(get_secret('PINECONE_API_KEY'),'nested-key')
            secrets.pop('pinecone')
            self.assertEqual(get_secret('PINECONE_API_KEY'),'env-key')

    def test_refreshed_openai_secret_overrides_stale_environment(self):
        with patch('agents.config.st.secrets',{'openai':{'api_key':'refreshed-key'}}), patch.dict(os.environ,{'OPENAI_API_KEY':'stale-key'}):
            configure_openai_credentials()
            self.assertEqual(os.environ['OPENAI_API_KEY'],'refreshed-key')

    def test_lazy_pinecone_uses_configured_index_namespace_and_embedding(self):
        secrets={'openai':{'api_key':'test-openai'},'pinecone':{'api_key':'test-pinecone',
            'index_name':'existing-index','namespace':'existing-documents','embedding_model':'text-embedding-3-large',
            'embedding_dimensions':1024,'text_key':'content','host':'fixture.example'}}
        with patch('agents.config.st.secrets',secrets), patch.dict(os.environ,{},clear=True), patch('agents.analysis_service.OpenAIEmbeddings') as embedding, patch('agents.analysis_service.PineconeVectorStore') as store:
            documents=DocumentSearch()
            store.assert_not_called()
            store.return_value.as_retriever.return_value.invoke.return_value=[Document(page_content='Verified document',metadata={'source':'fixture.pdf','page':3})]
            result=documents.search('Question')
            self.assertEqual(store.call_args.kwargs['pinecone_api_key'],'test-pinecone')
            self.assertEqual(store.call_args.kwargs['index_name'],'existing-index')
            self.assertEqual(store.call_args.kwargs['namespace'],'existing-documents')
            self.assertEqual(store.call_args.kwargs['text_key'],'content')
            self.assertEqual(store.call_args.kwargs['host'],'fixture.example')
            self.assertEqual(embedding.call_args.kwargs['dimensions'],1024)
            self.assertEqual(embedding.call_args.kwargs['model'],'text-embedding-3-large')
            self.assertNotIn('PINECONE_API_KEY',os.environ)
            self.assertEqual(result['rows'][0][:3],['D1','fixture.pdf',3])
            documents.search('Follow-up')
            self.assertEqual(store.call_count,1)

    def test_invalid_embedding_dimensions_fail_before_network(self):
        for dimensions in ['zero',0,-1]:
            with self.subTest(dimensions=dimensions), patch('agents.config.st.secrets',{'PINECONE_API_KEY':'test-key','PINECONE_EMBEDDING_DIMENSIONS':dimensions}), patch('agents.analysis_service.PineconeVectorStore') as store, patch.dict(os.environ,{},clear=True):
                with self.assertRaises(ValueError):
                    DocumentSearch().search('Question')
                store.assert_not_called()
