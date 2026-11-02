# Search timing

Search POST requests emit content-free logs from `mumucare.search_timing`,
correlated by the response's `X-Request-ID` / `request_id`.

Stages include employee authentication/rate limiting, Firestore context and
de-identification, query planning, cold model loading, Chroma collection opening,
query embedding, vector query, reranker prediction, each retrieval query,
candidate aggregation and response construction/de-identification. Completion
and failure events contain `elapsed_ms`; start events identify a stalled stage.
`search_results` reports short-term and strategy result counts, not their content.

`api_until_response_headers` measures from application dispatch (after reading
the body) to response headers, including validation and response serialization.
Its completed event does not imply HTTP success; check `search_response` status.
Nested timings overlap and must not be added to their parent totals.

This is not browser end-to-end latency. Cloud Run request latency also includes
platform overhead, and browser Network timing includes network transfer and
client-side waiting. Use these alongside the matching request ID. No actual
browser receipt or rendering can be measured by backend logs alone.

Example Cloud Logging filter after deploying this version:

```
resource.type="cloud_run_revision"
resource.labels.service_name="mumucare-llm"
"YOUR_REQUEST_ID"
```

Model-load events appear only on cache misses. Cold instances and warm instances
should be compared separately. Timing instrumentation does not change search
ranking, call external services for benchmarking, or deploy automatically.
