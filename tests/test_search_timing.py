import unittest

import search_timing


class SearchTimingTests(unittest.TestCase):
    def test_request_correlation_and_success(self):
        token = search_timing.request_id.set("test-request")
        self.addCleanup(search_timing.request_id.reset, token)
        with self.assertLogs(search_timing.logger, level="INFO") as logs:
            with search_timing.stage("test", candidate_count=6):
                pass
        self.assertEqual(len(logs.output), 2)
        self.assertIn("test-request", logs.output[1])
        self.assertIn("elapsed_ms", logs.output[1])

    def test_failure_reraises_without_logging_error_content(self):
        token = search_timing.request_id.set("test-request")
        self.addCleanup(search_timing.request_id.reset, token)
        with self.assertLogs(search_timing.logger, level="INFO") as logs:
            with self.assertRaises(ValueError):
                with search_timing.stage("test"):
                    raise ValueError("private content")
        self.assertIn("search_stage_failed", logs.output[-1])
        self.assertNotIn("private content", str(logs.output))

    def test_no_logs_outside_search_context(self):
        with self.assertNoLogs(search_timing.logger, level="INFO"):
            with search_timing.stage("test"):
                pass

    def test_decorator_preserves_return_value(self):
        @search_timing.timed("test")
        def example(value):
            return value
        self.assertEqual(example(42), 42)
