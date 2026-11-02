import unittest
from unittest.mock import patch

from langchain_core.documents import Document
import ifsp_isp_ai as core


class ConsolidatedRetrievalTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(core, "RERANKER_ENABLED", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_disabled_skips_model_and_keeps_vector_candidates(self):
        with patch.object(core, "RERANKER_ENABLED", False):
            docs, mock = self.run_search([[self.doc("one", 0.2)] for _ in range(6)])
        mock.assert_not_called()
        self.assertEqual(len(docs), 1)
        self.assertFalse(docs[0].metadata["reranker_enabled"])

    def run_search(self, results, score=2.0, topk=6):
        def rerank(query, docs, topk):
            self.assertIn("selected goal", query)
            for doc in docs:
                doc.metadata["rerank_score"] = score
            return docs[:topk]

        with patch.object(core, "retrieve_hybrid_with_meta", side_effect=results), \
             patch.object(core, "rerank_docs", side_effect=rerank) as mock:
            _, docs = core.retrieve_candidates(
                "身體福祉", "策略", "", ["selected goal"], topk=topk,
                plan={"search_queries": ["q1", "q2", "q3", "q4"]},
            )
        return docs, mock

    @staticmethod
    def doc(key, semantic=0.8):
        return Document(page_content="same text", metadata={"chroma_id": str(key),
                        "semantic_score": semantic})

    def test_single_rerank_and_candidate_cap(self):
        results = [[self.doc(i * 6 + j) for j in range(6)] for i in range(6)]
        docs, mock = self.run_search(results)
        mock.assert_called_once()
        self.assertEqual(len(mock.call_args.args[1]), 20)
        self.assertEqual(len(docs), 6)
        self.assertTrue(all("final_score" in doc.metadata for doc in docs))

    def test_deduplicates_by_id_and_keeps_query_matches(self):
        docs, mock = self.run_search([[self.doc("one")] for _ in range(6)])
        self.assertEqual(len(mock.call_args.args[1]), 1)
        self.assertEqual(len(docs), 1)
        self.assertIn("planner", docs[0].metadata["matched_tags"])
        self.assertIn("full", docs[0].metadata["matched_tags"])

    def test_empty_results(self):
        docs, mock = self.run_search([[] for _ in range(6)])
        self.assertEqual(docs, [])
        self.assertEqual(mock.call_args.args[1], [])

    def test_threshold_is_applied_after_rerank(self):
        results = lambda: [[self.doc("one", 0.2)] for _ in range(6)]
        docs, _ = self.run_search(results(), score=0.1)
        self.assertEqual(docs, [])
        docs, _ = self.run_search(results(), score=2)
        self.assertEqual(len(docs), 1)
