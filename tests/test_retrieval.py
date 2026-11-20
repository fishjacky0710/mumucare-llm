import unittest
from itertools import cycle
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

        with patch.object(core, "retrieve_hybrid_with_meta", side_effect=cycle(results)), \
             patch.object(core, "replace_name_with_stars", side_effect=lambda text: text), \
             patch.object(core, "rerank_docs", side_effect=rerank) as mock:
            _, docs = core.retrieve_candidates(
                "身體福祉", "策略", "", ["selected goal"], topk=topk,
                plan={"search_queries": ["q1", "q2", "q3", "q4"]},
            )
        return docs, mock

    @staticmethod
    def doc(key, semantic=0.8):
        return Document(page_content=f"reference {key}", metadata={"chroma_id": str(key),
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

    def test_display_duplicates_keep_best_original_id(self):
        best = self.doc("best", 0.9)
        best.page_content = "領域：A 內容：健康 生活。"
        duplicate = self.doc("duplicate", 0.8)
        duplicate.page_content = "領域：B 內容:健康生活"
        extra = [self.doc(f"extra-{i}", 0.7) for i in range(5)]
        with patch.object(core, "RERANKER_ENABLED", False):
            docs, _ = self.run_search([[duplicate, best] + extra for _ in range(6)])
        self.assertEqual(len(docs), 6)
        ids = [doc.metadata["chroma_id"] for doc in docs]
        self.assertIn("best", ids)
        self.assertNotIn("duplicate", ids)

    def test_normalizes_width_whitespace_and_trailing_punctuation(self):
        self.assertEqual(core.reference_text_key("ＡＢＣ 健康。"),
                         core.reference_text_key("ABC\n健康"))
        self.assertNotEqual(core.reference_text_key("同儕陪同"),
                            core.reference_text_key("家人陪同"))

    def test_expands_search_when_duplicates_fill_first_page(self):
        def retrieve(**kwargs):
            if kwargs["topk"] == 6:
                return [self.doc("duplicate")]
            return [self.doc(i) for i in range(6)]
        with patch.object(core, "RERANKER_ENABLED", False), \
             patch.object(core, "replace_name_with_stars", side_effect=lambda text: text), \
             patch.object(core, "retrieve_hybrid_with_meta", side_effect=retrieve) as search, \
             patch.object(core, "rerank_docs") as rerank:
            _, docs = core.retrieve_candidates("身體福祉", "策略", "", ["goal"],
                                              plan={"search_queries": []})
        self.assertEqual(len(docs), 6)
        self.assertEqual([call.kwargs["topk"] for call in search.call_args_list], [6, 6, 12, 12])
        rerank.assert_not_called()

    def test_ignores_internal_punctuation(self):
        self.assertEqual(core.reference_text_key("獲得體能活動、平衡訓練和步態訓練。"),
                         core.reference_text_key("獲得體能活動，平衡訓練和步態訓練"))
        self.assertEqual(core.reference_text_key('支持：「家人」（同儕）；陪伴！'),
                         core.reference_text_key("支持家人同儕陪伴"))
        self.assertNotEqual(core.reference_text_key("可獨立完成"),
                            core.reference_text_key("不可獨立完成"))

    def test_punctuation_variants_return_only_best_id(self):
        best = self.doc("best", 0.9)
        best.page_content = "體能活動、平衡訓練。"
        duplicate = self.doc("duplicate", 0.8)
        duplicate.page_content = "體能活動，平衡訓練"
        with patch.object(core, "RERANKER_ENABLED", False):
            docs, _ = self.run_search([[best, duplicate] for _ in range(6)])
        self.assertEqual([doc.metadata["chroma_id"] for doc in docs], ["best"])
