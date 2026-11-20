"""Content-free AI usage events and a summary of Cloud Logging JSON on stdin."""

from collections import Counter
from datetime import datetime
import hashlib
import json
import logging
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo


OPERATIONS = {
    "/v1/long-term-goals": "long_term_generate",
    "/v1/references/search": "reference_search",
    "/v1/goals/generate": "goals_generate",
    "/v1/goals/refine": "goals_refine",
}
logger = logging.getLogger("mumucare.usage")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.propagate = False


def record(operation, request_id, uid, status, elapsed_ms):
    event = {
        "event": "ai_usage", "event_id": str(uuid4()), "request_id": request_id,
        "operation": operation, "status": status, "success": 200 <= status < 300,
        "elapsed_ms": round(elapsed_ms, 2),
        "occurred_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "user_key": hashlib.sha256(("mumucare-usage-v1:" + uid).encode()).hexdigest() if uid else None,
    }
    logger.info(json.dumps(event))


def summarize(entries):
    events = {}
    for entry in entries:
        event = entry.get("jsonPayload", {})
        if event.get("event") == "ai_usage" and event.get("event_id"):
            events[event["event_id"]] = event
    rows = list(events.values())
    authenticated = [row for row in rows if row.get("user_key")]
    durations = sorted(row["elapsed_ms"] for row in authenticated)
    operation_stats = {}
    for operation in dict.fromkeys([*OPERATIONS.values(), *(row["operation"] for row in authenticated)]):
        operation_rows = [row for row in authenticated if row["operation"] == operation]
        operation_stats[operation] = {
            "requests": len(operation_rows),
            "unique_employees": len({row["user_key"] for row in operation_rows}),
            "successful_requests": sum(row["success"] for row in operation_rows),
            "failed_requests": sum(not row["success"] for row in operation_rows),
        }
    return {
        "authenticated_requests": len(authenticated),
        "unique_active_employees": len({row["user_key"] for row in authenticated}),
        "operations": dict(Counter(row["operation"] for row in authenticated)),
        "operation_stats": operation_stats,
        "successful_requests": sum(row["success"] for row in authenticated),
        "failed_requests": sum(not row["success"] for row in authenticated),
        "unauthenticated_requests": len(rows) - len(authenticated),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "mean_seconds": round(sum(durations) / len(durations) / 1000, 3) if durations else None,
        "p95_seconds": round(durations[max(0, (95 * len(durations) + 99) // 100 - 1)] / 1000, 3) if durations else None,
    }


if __name__ == "__main__":
    print(json.dumps(summarize(json.load(sys.stdin)), indent=2))
