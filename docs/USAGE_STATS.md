# AI Usage Statistics

No frontend changes, BigQuery export or extra model requests are needed.
After deployment, each completed AI POST emits one `ai_usage` JSON event.
The four operations are long-term generation, reference search, final generation
and regeneration. Context reads, service-user lists and CORS preflights do not
count as AI usage. An authenticated request counts even if it fails; an
unauthenticated request is counted separately and excluded from active users.

Events include a unique server-generated event ID, request ID, operation, local
Taipei timestamp, HTTP application status, elapsed milliseconds and a SHA-256
pseudonym derived from the Firebase UID. No UID, employee code, name, student ID,
ISP ID, prompt, generation content, authorization token or API key is logged.
Pseudonyms are still linkable user data; restrict log access accordingly.

## View Events

In Google Cloud Logging's Logs Explorer, select project `mumucare-llm`, choose
the desired time range and enter:

```text
resource.type="cloud_run_revision"
resource.labels.service_name="mumucare-llm"
jsonPayload.event="ai_usage"
```

Filter by `jsonPayload.operation="reference_search"` for one function, or by
`jsonPayload.success=false` for application failures. Events contain elapsed_ms,
not seconds.

## Summary in Terminal

From the project directory, run:

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="mumucare-llm" AND jsonPayload.event="ai_usage"' --project mumucare-llm --freshness=24h --limit=10000 --format=json | python3 usage_stats.py
```

The summary reports authenticated request count, distinct active employees,
counts by operation, successes/failures, unauthenticated requests, mean and P95
processing time in seconds. `operation_stats` additionally reports requests,
unique employees, successful requests and failed requests for each of the four
operations (including zero counts for unused operations). A person using both
search and generation counts once in each function, but only once globally;
per-operation unique users must not be summed to obtain total unique users.
These counts include authenticated failed attempts, not only successful calls.
This grouping works on existing `ai_usage` events and needs no frontend change
or redeployment of Cloud Run. Run the updated local summary command above;
Logs Explorer still displays individual events, not this computed summary.
Identical exported event IDs are deduplicated;
actual HTTP retries count separately. Inspect command failures before using the
summary; reaching 10,000 rows means the selected period may be truncated. Use a
smaller time range or raise the limit, and select a calendar-day timestamp range
when calendar-day statistics are required. `24h` is a rolling window.

## Limits

This records application processing until response headers, not browser display
or adoption of generated results. Unfinished requests and terminated processes
may have no completion event. A Cloud Run 504 does not stop application work, so
the application can log a later success after the platform already timed out;
use Cloud Run request logs as the source of truth for platform errors/latency.
This first version does not record model token counts or estimate costs.
Events are available only after deploying this code, not retroactively.
