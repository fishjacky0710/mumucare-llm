"use strict";

const OPERATION_NAMES = {
  long_term_generate: "生成長程目標",
  long_term_refine: "調整服務需求",
  reference_search: "搜尋參考資料",
  goals_generate: "生成短程目標與策略",
  goals_refine: "重新生成",
};

function taipeiDay(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return null;
  const parts = new Intl.DateTimeFormat("en", {
    timeZone: "Asia/Taipei", year: "numeric", month: "2-digit", day: "2-digit",
  }).formatToParts(date);
  const part = (name) => parts.find((item) => item.type === name).value;
  return `${part("year")}-${part("month")}-${part("day")}`;
}

function summarizeReport(entries, from = "", to = "") {
  if (!Array.isArray(entries)) throw new Error("請匯入 Logs Explorer 下載的 JSON 陣列。");
  if (from && to && from > to) throw new Error("開始日期不能晚於結束日期。");
  const unique = new Map();
  let invalid = 0;
  for (const entry of entries) {
    const event = entry && entry.jsonPayload;
    if (!event || event.event !== "ai_usage") continue;
    const day = taipeiDay(event.occurred_at || entry.timestamp);
    if (!event.event_id || !day || typeof event.operation !== "string" ||
        typeof event.success !== "boolean" || !Number.isFinite(event.elapsed_ms) ||
        event.elapsed_ms < 0 || (event.user_key != null && typeof event.user_key !== "string")) {
      invalid++;
      continue;
    }
    unique.set(event.event_id, { ...event, day });
  }
  const rows = [...unique.values()].filter((row) => (!from || row.day >= from) && (!to || row.day <= to));
  const authenticated = rows.filter((row) => row.user_key);
  function stats(group) {
    const times = group.map((row) => row.elapsed_ms).sort((a, b) => a - b);
    return {
      requests: group.length, users: new Set(group.map((row) => row.user_key)).size,
      success: group.filter((row) => row.success).length,
      failed: group.filter((row) => !row.success).length,
      mean: group.length ? times.reduce((a, b) => a + b, 0) / group.length / 1000 : null,
      p95: group.length ? times[Math.ceil(group.length * 0.95) - 1] / 1000 : null,
    };
  }
  const operationKeys = [...new Set([...Object.keys(OPERATION_NAMES), ...authenticated.map((row) => row.operation)])];
  return {
    ...stats(authenticated), invalid, excluded: rows.length - authenticated.length,
    imported: unique.size,
    operations: operationKeys.map((key) => ({
      name: OPERATION_NAMES[key] || "其他功能", ...stats(authenticated.filter((row) => row.operation === key)),
    })),
    days: [...new Set(authenticated.map((row) => row.day))].sort().map((day) => ({
      name: day, ...stats(authenticated.filter((row) => row.day === day)),
    })),
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { summarizeReport, taipeiDay };
}

if (typeof document !== "undefined") {
  let entries = [];
  let filename = "";
  let loadVersion = 0;
  const el = (id) => document.getElementById(id);
  const seconds = (value) => value === null ? "—" : value.toFixed(2);
  function chart(id, rows, field) {
    const target = el(id);
    target.replaceChildren();
    const max = Math.max(1, ...rows.map((row) => row[field]));
    for (const row of rows) {
      const container = document.createElement("div");
      container.className = "bar-row";
      const label = document.createElement("div");
      label.className = "bar-label";
      const name = document.createElement("span");
      name.textContent = row.name;
      const count = document.createElement("span");
      count.textContent = row[field].toLocaleString("zh-TW");
      label.append(name, count);
      const track = document.createElement("div");
      track.className = "track";
      const bar = document.createElement("div");
      bar.className = "fill";
      bar.style.width = `${row[field] / max * 100}%`;
      track.append(bar);
      container.append(label, track);
      target.append(container);
    }
  }
  function table(id, rows, withMean) {
    el(id).replaceChildren();
    for (const row of rows) {
      const tr = document.createElement("tr");
      const values = [row.name, row.requests, row.users, row.success, row.failed];
      if (withMean) values.push(seconds(row.mean));
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        tr.append(cell);
      }
      el(id).append(tr);
    }
  }
  function render() {
    try {
      const report = summarizeReport(entries, el("from").value, el("to").value);
      el("report").hidden = false;
      el("empty").hidden = true;
      el("requests").textContent = report.requests.toLocaleString("zh-TW");
      el("users").textContent = report.users.toLocaleString("zh-TW");
      el("success").textContent = report.requests ? `${(report.success / report.requests * 100).toFixed(1)}%` : "—";
      el("p95").textContent = seconds(report.p95);
      chart("operation-chart", report.operations, "requests");
      chart("people-chart", report.operations, "users");
      chart("daily-chart", report.days, "requests");
      table("operations", report.operations, true);
      table("days", report.days, false);
      el("excluded").textContent = `本期間未登入紀錄 ${report.excluded} 筆，不計入使用人數；格式異常紀錄 ${report.invalid} 筆已略過。`;
      el("status").className = "status";
      el("status").textContent = `${filename} · 已匯入 ${report.imported} 筆不重複統計事件 · ${report.days.length ? report.days[0].name + " 至 " + report.days.at(-1).name : "目前範圍無已驗證操作"}`;
    } catch (error) {
      el("report").hidden = true;
      el("status").className = "status error";
      el("status").textContent = error.message;
    }
  }
  el("file").addEventListener("change", async (event) => {
    const version = ++loadVersion;
    const file = event.target.files[0];
    if (!file) return;
    el("report").hidden = true;
    el("empty").hidden = true;
    el("status").textContent = "正在讀取…";
    try {
      const parsed = JSON.parse(await file.text());
      if (version !== loadVersion) return;
      if (!Array.isArray(parsed)) throw new Error("請選擇 Logs Explorer 下載的 JSON 陣列檔案。");
      entries = parsed;
      filename = file.name;
      el("from").value = "";
      el("to").value = "";
      render();
    } catch (error) {
      if (version !== loadVersion) return;
      entries = [];
      filename = "";
      el("status").className = "status error";
      el("status").textContent = `無法讀取檔案：${error.message}`;
    }
  });
  for (const id of ["from", "to"]) el(id).addEventListener("change", () => { if (filename) render(); });
  el("reset").addEventListener("click", () => {
    el("from").value = "";
    el("to").value = "";
    if (filename) render();
  });
}
