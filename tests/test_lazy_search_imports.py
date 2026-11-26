import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ifsp_isp_ai as core


class LazySearchImportTests(unittest.TestCase):
    def test_embedder_loads_on_demand_and_is_cached(self):
        constructor = Mock()
        core.get_embedder.cache_clear()
        self.addCleanup(core.get_embedder.cache_clear)
        with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(
            SentenceTransformer=constructor
        )}):
            first = core.get_embedder()
            self.assertIs(core.get_embedder(), first)
        constructor.assert_called_once_with(core.EMBEDDING_MODEL)

    def test_collection_opens_on_demand_and_is_cached(self):
        client = Mock()
        core.get_chroma_collection.cache_clear()
        self.addCleanup(core.get_chroma_collection.cache_clear)
        with patch.dict(sys.modules, {"chromadb": SimpleNamespace(PersistentClient=client)}):
            first = core.get_chroma_collection()
            self.assertIs(core.get_chroma_collection(), first)
        client.assert_called_once_with(path=core.DBDIR)
        client.return_value.get_collection.assert_called_once_with(core.COLLECTION)

    def test_backend_starts_without_search_dependencies(self):
        # A fresh interpreter catches transitive imports that a warmed test process hides.
        result = subprocess.run(
            [sys.executable, "-c", """
import importlib.abc
import sys

blocked = {'chromadb', 'sentence_transformers', 'transformers', 'torch'}
class BlockSearchImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Unexpected search dependency: ' + fullname)

sys.meta_path.insert(0, BlockSearchImports())
import cloud_run_backend
assert not blocked.intersection(sys.modules)
assert '/v1/long-term-goals' in {route.path for route in cloud_run_backend.app.routes}
"""],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
