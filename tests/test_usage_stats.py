import json
import unittest
from unittest.mock import patch

import usage_stats


class UsageStatsTests(unittest.TestCase):
    def test_api_emits_one_summary_and_ignores_preflight(self):
        from fastapi.testclient import TestClient
        from cloud_run_backend import app

        with patch.object(usage_stats.logger, "info") as log:
            with TestClient(app) as client:
                client.options("/v1/references/search")
                response = client.post("/v1/references/search", json={})
        log.assert_called_once()
        event = json.loads(log.call_args.args[0])
        self.assertEqual(response.status_code, 401)
        self.assertEqual(event["status"], 401)
        self.assertEqual(event["request_id"], response.headers["X-Request-ID"])
        self.assertIsNone(event["user_key"])

    def test_privacy_and_summary(self):
        with patch.object(usage_stats.logger, "info") as log:
            usage_stats.record("reference_search", "request-1", "private-uid", 200, 1200)
            event = json.loads(log.call_args.args[0])
        self.assertNotIn("private-uid", json.dumps(event))
        self.assertEqual(event["elapsed_ms"], 1200)
        entries = [{"jsonPayload": event}, {"jsonPayload": event}]
        result = usage_stats.summarize(entries)
        self.assertEqual(result["authenticated_requests"], 1)
        self.assertEqual(result["unique_active_employees"], 1)
        self.assertEqual(result["mean_seconds"], 1.2)

    def test_unique_users_per_operation_and_global_deduplication(self):
        def entry(identifier, operation, user, success=True):
            return {"jsonPayload": {"event": "ai_usage", "event_id": identifier,
                    "user_key": user, "status": 200 if success else 401,
                    "success": success, "operation": operation, "elapsed_ms": 100}}
        search = entry("1", "reference_search", "user-a")
        entries = [search, search, entry("2", "reference_search", "user-a"),
                   entry("3", "reference_search", "user-b", False),
                   entry("4", "goals_generate", "user-a"),
                   entry("5", "reference_search", None, False)]
        result = usage_stats.summarize(entries)
        self.assertEqual(result["unique_active_employees"], 2)
        self.assertEqual(result["operation_stats"]["reference_search"], {
            "requests": 3, "unique_employees": 2,
            "successful_requests": 2, "failed_requests": 1})
        self.assertEqual(result["operation_stats"]["goals_generate"]["unique_employees"], 1)
        self.assertEqual(result["operation_stats"]["goals_refine"]["requests"], 0)

    def test_empty_period_has_zero_counts_for_all_operations(self):
        result = usage_stats.summarize([])
        self.assertEqual(len(result["operation_stats"]), 4)
        self.assertTrue(all(stats["unique_employees"] == 0
                            for stats in result["operation_stats"].values()))
    def test_unauthenticated_requests_excluded_from_usage(self):
        event = {"event": "ai_usage", "event_id": "1", "user_key": None,
                 "status": 401, "success": False, "operation": "reference_search", "elapsed_ms": 10}
        result = usage_stats.summarize([{"jsonPayload": event}])
        self.assertEqual(result["authenticated_requests"], 0)
        self.assertEqual(result["unauthenticated_requests"], 1)
        self.assertIsNone(result["p95_seconds"])

    def test_distinct_requests_count_separately_for_same_user(self):
        entries = [{"jsonPayload": {"event": "ai_usage", "event_id": str(i),
                    "user_key": "one-user", "status": 200, "success": True,
                    "operation": "goals_generate", "elapsed_ms": 1000}} for i in range(2)]
        result = usage_stats.summarize(entries)
        self.assertEqual(result["authenticated_requests"], 2)
        self.assertEqual(result["unique_active_employees"], 1)
