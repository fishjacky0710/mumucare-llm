import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from long_term_refinement import parse_result, refine_long_term_goals


class RefinementTests(unittest.TestCase):
    def test_edits_and_instruction_are_preserved(self):
        invoke = Mock(return_value='{"important_to":[" [參與]活動以[提升]生活品質。 "],"important_for":[]}')
        result = refine_long_term_goals(domain="身體福祉", context={"summary": "現況"},
            important_to=["手動編輯"], important_for=["支持"], instruction="簡化", invoke=invoke)
        self.assertEqual(result["important_to"], ["[參與]活動以[提升]生活品質。"])
        payload = json.loads(invoke.call_args.args[0][1][1])
        self.assertEqual(payload["original_important_to"], ["手動編輯"])
        self.assertEqual(payload["refine_prompt"], "簡化")

    def test_rejects_invalid_results(self):
        for result in ['{}', '[]', '{"important_to":[],"important_for":[]}',
                       '{"important_to":[" "],"important_for":[]}',
                       '{"important_to":[1],"important_for":[]}',
                       '{"result":"short term plan"}', 'not json']:
            with self.subTest(result=result), self.assertRaises(ValueError):
                parse_result(result)

    def test_accepts_json_code_fence(self):
        self.assertEqual(parse_result('```json\n{"important_to":["[參與]活動以[提升]生活品質。"],"important_for":[]}\n```')["important_to"], ["[參與]活動以[提升]生活品質。"])

    def test_missing_brackets_are_repaired_once(self):
        valid = '{"important_to":["[參與]活動以[提升]生活品質。"],"important_for":[]}'
        invoke = Mock(side_effect=['{"important_to":["參與活動以提升生活品質。"],"important_for":[]}', valid])
        result = refine_long_term_goals(domain="身體福祉", context={}, important_to=["原需求"],
            important_for=[], instruction="簡化", invoke=invoke)
        self.assertEqual(result["important_to"], ["[參與]活動以[提升]生活品質。"])
        self.assertEqual(invoke.call_count, 2)
        self.assertIn("不得省略 []", invoke.call_args.args[0][0][1])

    def test_invalid_brackets_never_return_as_success(self):
        for goal in ["參與活動以提升生活品質。", "[參與]活動以提升生活品質。", "[參與活動以提升生活品質。]"]:
            invoke = Mock(return_value=json.dumps({"important_to": [goal], "important_for": []}))
            with self.subTest(goal=goal), self.assertRaises(ValueError):
                refine_long_term_goals(domain="身體福祉", context={}, important_to=["原需求"],
                    important_for=[], instruction="簡化", invoke=invoke)
            self.assertEqual(invoke.call_count, 2)


class EndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the actual routes with model/retrieval dependencies replaced; no cloud calls.
        core = Mock()
        core.replace_name_with_stars.side_effect = lambda value: value
        core.mask_known_names.side_effect = lambda value, names: value
        path = Path(__file__).resolve().parents[1] / "cloud_run_backend.py"
        spec = importlib.util.spec_from_file_location("backend_refinement_test", path)
        cls.backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.backend
        with patch.dict(sys.modules, {"ifsp_isp_ai": core}), patch.dict(os.environ, {"APP_ENV": "development"}):
            spec.loader.exec_module(cls.backend)
        cls.backend.app.dependency_overrides[cls.backend.verify_employee] = lambda: Mock()
        cls.client = TestClient(cls.backend.app)

    def setUp(self):
        self.payload = {"studentId": "s", "ispId": "i", "domain": "身體福祉",
            "original_important_to": ["原需求"], "original_important_for": [], "refine_prompt": "簡化"}

    def test_success_and_authorization_context(self):
        with patch.object(self.backend, "load_adult_context", return_value={}) as context, patch.object(
            self.backend, "refine_long_term_goals", return_value={"important_to": ["[參與]活動以[提升]生活品質。"], "important_for": []}
        ):
            response = self.client.post('/v1/long-term-goals/refine', json=self.payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["important_to"], ["[參與]活動以[提升]生活品質。"])
        self.assertTrue(response.json()["request_id"])
        context.assert_called_once()

    def test_denied_context_never_calls_model(self):
        with patch.object(self.backend, "load_adult_context", side_effect=HTTPException(403)), patch.object(
            self.backend, "refine_long_term_goals"
        ) as generate:
            response = self.client.post('/v1/long-term-goals/refine', json=self.payload)
        self.assertEqual(response.status_code, 403)
        generate.assert_not_called()

    def test_blank_prompt_rejected(self):
        with patch.object(self.backend, "load_adult_context", return_value={}):
            response = self.client.post('/v1/long-term-goals/refine', json={**self.payload, 'refine_prompt': ' '})
        self.assertIn(response.status_code, (400, 422))

    def test_invalid_model_result_is_service_error(self):
        with patch.object(self.backend, "load_adult_context", return_value={}), patch.object(
            self.backend, "refine_long_term_goals", side_effect=ValueError("invalid format")
        ):
            response = self.client.post('/v1/long-term-goals/refine', json=self.payload)
        self.assertEqual(response.status_code, 503)

    def test_oversized_prompt_rejected(self):
        response = self.client.post('/v1/long-term-goals/refine', json={**self.payload, 'refine_prompt': 'x' * 5001})
        self.assertEqual(response.status_code, 422)

    def test_reference_generation_all_four_combinations(self):
        for short, strategy, builder in [([], [], "build_case3_prompt"),
                (["short"], [], "build_case1_prompt"), ([], ["strategy"], "build_strategy_only_prompt"),
                (["short"], ["strategy"], "build_case2_prompt")]:
            with self.subTest(short=short, strategy=strategy), patch.object(
                self.backend, "load_adult_context", return_value={"basic_info": "個案資訊"}
            ), patch.object(self.backend, "load_reference_texts", side_effect=lambda ids, *_: ids):
                core = self.backend.core
                core.reset_mock()
                core.build_selected_context.side_effect = lambda values: "\n".join(values)
                core.get_llm.return_value.invoke.return_value.content = "生成結果"
                response = self.client.post('/v1/goals/generate', json={
                    "studentId": "s", "ispId": "i", "domain": "身體福祉", "selected_long_goal": "長程目標",
                    "selected_short_reference_ids": short, "selected_strategy_reference_ids": strategy})
                self.assertEqual(response.status_code, 200, response.text)
                getattr(core, builder).assert_called_once()
                if strategy:
                    self.assertEqual(getattr(core, builder).call_args.args[-1], "strategy")

    def test_strategy_only_prompt_includes_reference_without_no_selection_claim(self):
        import prompt
        messages = prompt.build_case3_prompt("身體福祉", "個案資訊與長程目標", "安排可選擇的活動")
        text = "\n".join(content for _, content in messages)
        self.assertIn("安排可選擇的活動", text)
        self.assertIn("使用者僅勾選策略參考", text)
        self.assertNotIn("使用者沒有勾選任何短程目標與策略", text)
