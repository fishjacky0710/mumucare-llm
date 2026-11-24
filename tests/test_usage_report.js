"use strict";
const assert = require("node:assert/strict");
const { summarizeReport, taipeiDay } = require("../usage_report.js");
const entry = (id, user, operation, success = true, date = "2026-11-20T23:00:00+08:00") => ({
  jsonPayload: { event: "ai_usage", event_id: id, user_key: user, operation,
    success, elapsed_ms: 1000, occurred_at: date },
});
const first = entry("1", "u1", "reference_search");
const rows = [first, first, entry("2", "u1", "reference_search"),
  entry("3", "u2", "goals_generate", false), entry("4", null, "reference_search", false)];
const report = summarizeReport(rows);
assert.equal(report.requests, 3);
assert.equal(report.users, 2);
assert.equal(report.excluded, 1);
assert.equal(report.operations.find((item) => item.name === "搜尋參考資料").users, 1);
assert.equal(report.operations.find((item) => item.name === "搜尋參考資料").requests, 2);
assert.equal(report.failed, 1);
assert.equal(report.p95, 1);
assert.equal(report.days[0].users, 2);
assert.equal(taipeiDay("2026-11-20T17:00:00Z"), "2026-11-21");
assert.equal(summarizeReport(rows, "2026-11-21").requests, 0);
assert.equal(summarizeReport([]).p95, null);
assert.throws(() => summarizeReport({}), /JSON/);
assert.throws(() => summarizeReport([], "2026-11-22", "2026-11-20"), /開始日期/);
assert.equal(summarizeReport([{ jsonPayload: { event: "ai_usage" } }]).invalid, 1);
console.log("Local report tests passed");

const refined = summarizeReport([entry("refine-1", "u1", "long_term_refine")]);
assert.equal(refined.operations.find((item) => item.name === "調整服務需求").requests, 1);
