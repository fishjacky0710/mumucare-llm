# 服務需求補充指示 API

前端第一次生成服務需求後顯示「補充指示」輸入框。送出完整目前結果（包含手動編輯、未勾選項目），成功後更新小幫手結果並清空勾選；不直接寫入 ISP。失敗保留原結果與選取。

POST `/v1/long-term-goals/refine`

沿用員工 Firebase Bearer token、個案權限驗證、CORS、速率限制與 X-Request-ID。

```json
{
  "studentId": "student-id",
  "ispId": "isp-id",
  "domain": "身體福祉",
  "service_type": "成人",
  "original_important_to": ["目前對他重要的需求"],
  "original_important_for": ["目前為他重要的需求"],
  "refine_prompt": "請增加自主選擇，並簡化文字"
}
```

每個需求陣列最多 10 項、每項 1–6000 字，兩類不可同時為空。補充指示 1–5000 字，不可只有空白。

回應沿用長程目標格式：`request_id`、`important_to`、`important_for`、`mode`。模型格式不符回覆 503，前端保留原內容。

先部署 mumucare-llm 後端至 Cloud Run，再部署前端。既有 `/v1/long-term-goals` 與 `/v1/goals/refine` 不變。這次修改不會自行部署或調整 CORS。
