# Mumucare ISP AI API 串接文件

更新日期：2026-10-08，API v1.1。依據 `cloud_run_backend.py`、`employee_auth.py` 與 `firestore_repository.py` 整理。

## 1. 服務資訊與目前狀態

| 項目 | 內容 |
|---|---|
| Base URL | `https://mumucare-llm-663386326083.asia-east1.run.app` |
| GCP 專案 | `mumucare-llm` |
| 區域 | `asia-east1` |
| 支援服務類型 | 成人 |
| 請求格式 | POST 使用 JSON；編碼 UTF-8 |
| 業務 API 驗證 | `Authorization: Bearer <Firebase ID Token>`，Firebase 專案為 `mumu-care` |
| 處理模式 | 同步請求，完成後一次回傳；沒有串流或工作輪詢 API |
| 部署 request timeout | 300 秒 |
| 容量設定 | 每個 instance 1 個併發請求，最多 5 個 instances，最少 0 |
| Swagger / ReDoc / OpenAPI 網址 | 正式環境關閉 |

新版已部署 revision：`mumucare-llm-00002-f85`。已完成以下外部測試：

| API v1.1 測試 | 結果 |
|---|---|
| `GET /` | HTTP 200 |
| 只帶舊 `X-API-Key` | HTTP 401 |
| 帶無效 Bearer Token | HTTP 401 |
| `https://mumu-care.web.app` 的 Authorization CORS 預檢 | HTTP 200，正確允許來源及 Authorization header |

本機另有 12 個權限與驗證測試通過。真實員工 Token 的完整操作流程尚需前端驗證。

Firestore 通道已在前一版完成跨專案授權與實測：

| 實測項目 | 結果 |
|---|---|
| `GET /` | HTTP 200 |
| 舊版 `GET /v1/service-users` | HTTP 200，66 位符合條件 |
| 舊版 `POST /v1/isps` | HTTP 200，抽測服務使用者有 2 份 ISP |
| 舊版 `POST /v1/context` | HTTP 200，基本資料、現況摘要、夢想皆有內容 |

以上筆數是舊版測試當下結果，前端不可固定寫死。新版名單依員工角色過濾，筆數可能不同。
新版真實員工 Token 登入、RAG、OpenAI 初次生成與重新生成尚需前端端到端實測。

Cloud Run 使用以下服務帳號讀取 `mumu-care` 的 Firestore：

```text
mumucare-llm-runtime@mumucare-llm.iam.gserviceaccount.com
```

Firestore 所需 `roles/datastore.viewer` 已授權。新版撤銷／停用檢查另需在 `mumu-care` 授予 `roles/firebaseauth.viewer`；未授權時，有效 Token 的登入驗證可能回傳 503。

CORS allowlist 為 `https://mumu-care.web.app`、`https://mumu-care.firebaseapp.com` 與測試網址 `https://mumu-care--jacky-lryp8mk8.web.app`。前端可以直接呼叫 Cloud Run，不需要 Gateway。

## 2. 驗證與前端連線方式

所有 `/v1/*` API 都需要：

```http
Authorization: Bearer <目前登入員工的 Firebase ID Token>
Content-Type: application/json
X-Request-ID: <可選的追蹤 ID>
```

GET 不需要 `Content-Type`。`X-Request-ID` 可用 1–64 個英數字、`.`、`_`、`-`；未提供或不符合格式時，後端產生 UUID。
後端在回應 JSON 的 `request_id` 與回應 header 的 `X-Request-ID` 提供追蹤值。公開根路徑的 JSON 不包含 `request_id`。

目前流程：

```text
Angular → 帶 Firebase ID Token → 本 Cloud Run API → Firestore / RAG / OpenAI
```

後端會驗證：

- Token 由 Firebase `mumu-care` 發出，簽章正確且未過期、撤銷，帳號未停用。
- `email_verified == true`、信箱網域為 `maria.org.tw`、登入 provider 為 `google.com`。
- `users/{uid}.code` 與已驗證信箱推導的員工編號一致，且 profile 未標示停用／刪除。
- `employees` 中 `employeeId` 對應該編號，且 `jobStatus == 1`。
- `roles.userCodes` 包含該員工編號，並具備對應服務單位與領域角色。
- 個案符合在案、服務單位及 ISP 存在條件。

角色範圍：`admin` 可存取全部允許服務；`serviceIds` 的 `_ALL` 表示全部允許服務；其他情況依 `cases.service.id` 對照。
成人各領域允許 `manager`、`educareGiver`、`physicalTherapy`、`occupationalTherapy`、`speechTherapy`；身體福祉另允許 `nurse`、`nutritionist`。
只有 `socialWorker` 角色不能使用此版成人八大領域 AI API。

舊 `X-API-Key` 不再接受。前端不需要共用金鑰；Firebase Web 設定中的公開 `apiKey` 也不是這裡的登入憑證。
不要傳 Google OAuth access token 或 Firebase custom token；請使用登入中的 Firebase user 的 `getIdToken()`。

## 3. 給「個案管理」前端專案的實作要求

目標是整合現有成人 ISP 編輯頁，不需要另建一套登入、服務使用者或 ISP 管理頁。

| 層級 | 要實作的內容 |
|---|---|
| Angular API service | 七個業務 API 的 request / response 型別、Firebase ID Token、loading 與錯誤處理；正式環境直接使用 Cloud Run Base URL |
| Cloud Run | 驗證員工登入、在職狀態與個案／領域權限；已由此版後端實作 |
| 成人 ISP 編輯畫面 | 傳入目前 `studentId`、ISP document ID 與中文領域，按步驟顯示長程目標、RAG 參考與生成結果 |
| 儲存 | 使用原有 ISP 儲存流程；本後端不寫入生成結果 |

前端所有 POST 的 JSON body 與上一版一致，只需切換 API Base URL 與 Authorization header。
`POST /v1/isps` 與 `GET /v1/service-users` 依員工的可操作服務範圍查詢；帶 `domain` 的端點另檢查該領域角色。

對現有 Angular 專案的對接位置：

- `src/app/form/adult/isp/services/isp-ai-api.service.ts`：統一 API 呼叫。
- `src/environments/environment.prod.ts`：配置 Cloud Run Base URL，`useFirebaseAuth` 為 `true`；不配置共用 API key。
- `proxy.conf.json`：可保留本機轉送並攜帶 Authorization，部署後不會自動生效。
- 既有成人 ISP AI dialog：管理單選目標、參考資料 ID、完整結果與重新生成。

同目錄的 `isp-ai-api.types.ts` 提供可直接參考的 TypeScript 資料型別。

### 畫面狀態與操作

1. 進入成人 ISP 領域編輯頁時，沿用頁面已選的服務使用者、ISP、領域；不必重複列出選單。
2. 按下生成長程目標後才顯示 `important_to` 與 `important_for` 選項。
3. 兩組合計只能選一個目標；選定後顯示「搜尋參考資料」，不要在選取時自動搜尋。
4. 搜尋成功後，分開呈現短程目標與策略 checkbox 清單，並顯示 AI 生成按鈕。
5. 生成成功後，顯示完整 `result`，以及調整方向輸入框與「重新生成」按鈕。
6. 重新生成成功後以完整新 `result` 取代舊結果。
7. 按「重新選擇長程目標」時，解鎖並清空選取，清除參考資料、生成結果、調整文字與錯誤狀態，回到長程目標選擇階段。
8. 切換服務使用者、ISP 或領域時清空整個 AI 流程；處理中的舊請求即使完成，也不得更新新個案畫面。

AI 入口只應顯示於可編輯的成人八大領域；需符合現有角色與可編輯權限，以及後端目前的服務範圍。
兒童與社工不能使用此版 API。

### 呼叫順序

1. 選服務使用者：`GET /v1/service-users`，如果前端已有名單可略過。
2. 選 ISP：`POST /v1/isps`，如果前端已有 ISP 清單可略過。
3. 選領域、呈現現況：`POST /v1/context`，如果不需顯示摘要可略過。
4. 按生成長程目標：`POST /v1/long-term-goals`。
5. 使用者選定一個長程目標，按搜尋參考資料：`POST /v1/references/search`。
6. 使用者勾選參考資料、按 AI 生成：`POST /v1/goals/generate`。
7. 使用者輸入調整方向、按重新生成：`POST /v1/goals/refine`。

每次請求都要帶所需身分欄位。後端不保存前一個請求的選擇，不需要 session ID。
切換服務使用者、ISP 或領域時，前端應清空長程目標、參考資料、生成結果與修改文字；過時的請求回應不得覆蓋新選擇。

## 4. 共用欄位與支援範圍

除名單 API 外，多數 POST 使用以下欄位：

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| `studentId` | string | 是 | `students` document ID；1–200 字元，不可含 `/` |
| `ispId` | string | 是 | `students/{studentId}/forms/adult/isp/{ispId}` document ID；1–200 字元，不可含 `/` |
| `domain` | string | 是 | 下列八個領域之一；傳中文名稱 |
| `service_type` | string | 否 | 預設 `成人`，目前只能傳 `成人` |

`POST /v1/isps` 只需要 `studentId`。

| 領域名稱 | ISP `personalProfile` 欄位 |
|---|---|
| 身體福祉 | `physicalWellBeing` |
| 情緒福祉 | `emotionalWellBeing` |
| 物質福祉 | `materialWellBeing` |
| 個人發展 | `personalDevelopment` |
| 自我決策 | `selfDetermination` |
| 人際關係 | `interpersonalRelations` |
| 權利 | `rights` |
| 社會融合 | `socialInclusion` |

服務使用者必須在 `cases` 中有符合以下條件的紀錄：

```text
status == "enrollment"
service.name == "雲端發展室" 或 "技能養成第一中心"
student.id == 傳入的 studentId
```

沒有符合條件的服務使用者會收到 403。ISP 必須存在，且 `isDeleted` 不可為 `true`。
後端會同時依員工角色的 `serviceIds` 過濾服務使用者與驗證個案存取。

## 5. API 明細

以下 ID、名字、內容與分數皆為示例，不是實際資料，也不保證每次模型輸出相同。
所有成功的業務 API 回傳 HTTP 200。

### GET /

不需要金鑰，用於確認服務連線。

```json
{"service": "Mumucare ISP AI API"}
```

程式也宣告 `GET /healthz`，回應為 `{"status":"ok"}`；部署後的外部測試目前收到 Google 層的 404。
目前請使用 `GET /` 做外部連線檢查。它不會驗證 Firestore、ChromaDB 或 OpenAI 是否可用。

### GET /v1/service-users

取得符合在案與服務單位條件的服務使用者名單。沒有 request body，沒有分頁參數。

```json
{
  "request_id": "example-request-001",
  "service_users": [
    {"id": "student-example-001", "name": "示例服務使用者"}
  ]
}
```

名單的 `name` 保留姓名供選單使用，並非去識別化的文字。

### POST /v1/isps

取得服務使用者的成人 ISP 清單及基本資料，排除已刪除 ISP。

Request：

```json
{"studentId": "student-example-001"}
```

Response：

```json
{
  "request_id": "example-request-002",
  "basic_info": "障礙等級：中度\n障礙類別：第一類",
  "isps": [
    {"id": "isp-example-001", "name": "115年度個別服務計畫"}
  ]
}
```

`name` 取自 ISP 的 `friendlyName`，沒有名稱時使用 document ID。
清單按名稱、ID 反向排序，並非依建立日期排序。

### POST /v1/context

讀取所選 ISP、領域的基本資料、現況摘要與夢想，整理並去識別化後回傳。

Request：

```json
{
  "studentId": "student-example-001",
  "ispId": "isp-example-001",
  "domain": "身體福祉",
  "service_type": "成人"
}
```

Response：

```json
{
  "request_id": "example-request-003",
  "basic_info": "障礙等級：中度\n障礙類別：第一類",
  "summary": "【教保員摘述】\n**在支持下參與日常活動。",
  "dream": "希望能參與喜歡的休閒活動。"
}
```

來源：`basic_info` 是 `students/{studentId}.disabilityCard`；`dream` 是同份 ISP 的 `dream`。
`summary` 包含所選 `personalProfile` 領域，以及 `reference` 連結的相關專業評估摘要。
前端不需要另外送出這三個文字欄位。

### POST /v1/long-term-goals

後端自行讀取並去識別化個案資料，再生成「對他重要」與「為他重要」的長程目標選項。

Request：

```json
{
  "studentId": "student-example-001",
  "ispId": "isp-example-001",
  "domain": "身體福祉",
  "service_type": "成人",
  "custom_important_to": ["希望參與自己喜歡的戶外活動"],
  "custom_important_for": ["維持日常活動安全"]
}
```

| 額外欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| `custom_important_to` | string[] | 否 | 對他重要；預設 `[]`，最多 10 筆 |
| `custom_important_for` | string[] | 否 | 為他重要；預設 `[]`，最多 10 筆 |

Response：

```json
{
  "request_id": "example-request-004",
  "important_to": ["[參與]喜歡的戶外活動以[提升]生活滿意度。"],
  "important_for": ["[建立]安全活動習慣以[維持]日常參與。"],
  "mode": "live"
}
```

每組最多 5 個目標。使用者應從兩組中合計選一個，將完整句子保存為 `selected_long_goal`。
單選及選定後鎖定是前端行為；後端接收一個字串，不另驗證它是否出自上次生成。

### POST /v1/references/search

依領域、基本資料及選定長程目標，產生搜尋方向並查詢 ChromaDB，使用 embedding 與 reranker 篩選參考資料。

Request：

```json
{
  "studentId": "student-example-001",
  "ispId": "isp-example-001",
  "domain": "身體福祉",
  "service_type": "成人",
  "selected_long_goal": "[建立]安全活動習慣以[維持]日常參與。",
  "top_k": 6
}
```

| 額外欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| `selected_long_goal` | string | 是 | 完整長程目標句；1–3,000 字元 |
| `top_k` | integer | 否 | 預設 6，範圍 1–10；每類候選上限，不保證返回滿額 |

Response：

```json
{
  "request_id": "example-request-005",
  "search_direction": ["成人日常活動安全支持"],
  "short_term_references": [
    {"id": "short-ref-example-001", "text": "在支持下參與日常活動。", "score": 0.82, "kind": "short_term"}
  ],
  "strategy_references": [
    {"id": "strategy-ref-example-001", "text": "運用安全動線與同儕支持參與活動。", "score": 0.78, "kind": "strategy"}
  ],
  "mode": "live"
}
```

`id` 是後端參考資料的 Chroma ID，生成時回傳這個 ID，不要自行編號或只傳 `text`。
`score` 是 0–1 的相對排序分數，不是正確率或符合需求的機率。
兩類參考資料可能都是空陣列。前端應提供空結果狀態，仍可讓使用者生成。

### POST /v1/goals/generate

生成短程目標與支持策略。後端依參考資料 ID 重新讀取原文，前端只需要傳選中的 ID。

Request：

```json
{
  "studentId": "student-example-001",
  "ispId": "isp-example-001",
  "domain": "身體福祉",
  "service_type": "成人",
  "selected_long_goal": "[建立]安全活動習慣以[維持]日常參與。",
  "selected_short_reference_ids": ["short-ref-example-001"],
  "selected_strategy_reference_ids": ["strategy-ref-example-001"]
}
```

| 額外欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| `selected_long_goal` | string | 是 | 1–3,000 字元，應與搜尋時的選擇一致 |
| `selected_short_reference_ids` | string[] | 否 | 預設 `[]`，最多 10 筆 |
| `selected_strategy_reference_ids` | string[] | 否 | 預設 `[]`，最多 10 筆 |

Response：

```json
{
  "request_id": "example-request-006",
  "result": "【面向：自然支持】\n\n短程目標：\n1-1 ...\n\n支持策略：\n1-1-1 ...\n\n【面向：科技／輔具／工具】\n...",
  "mode": "live"
}
```

`result` 是完整文字，並非結構化 JSON；示例省略了其餘內容。
Prompt 要求成人七大支持面向，但此端點尚未以程式驗證七個面向是否完整。

參考資料 ID 必須存在，且符合目前領域及所傳類型，否則回傳 400。
目前的 prompt 分支為：

- 有短程目標、有策略參考：同時採用兩類。
- 只有短程目標參考：採用短程目標參考。
- 沒有短程目標參考：使用無參考的生成分支；即使只勾策略參考，目前也不會帶入該策略內容。

這是現行行為，前端需避免誤以為「只勾策略」也一定會被採用。

### POST /v1/goals/refine

使用者不滿意結果時，傳完整原始結果與調整方向，重新生成。

Request：

```json
{
  "studentId": "student-example-001",
  "ispId": "isp-example-001",
  "domain": "身體福祉",
  "service_type": "成人",
  "original": "【面向：自然支持】\n\n短程目標：\n1-1 ...\n\n支持策略：\n1-1-1 ...\n...完整七個面向內容...",
  "refine_prompt": "請增加多元支持方式，評量結尾依實際行為調整，重新生成全部七個面向。"
}
```

| 額外欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| `original` | string | 是 | 原本完整 `result`；1–60,000 字元 |
| `refine_prompt` | string | 是 | 調整方向；1–5,000 字元 |

Response：

```json
{
  "request_id": "example-request-007",
  "result": "...調整後的完整內容...",
  "mode": "live"
}
```

未指定面向時，規則要求重新生成全部七個面向。指定面向時，調整指定面向並保留其他面向；前端應用回傳的完整 `result` 取代原結果。
後端會檢查重新生成的面向與格式，必要時嘗試修復一次；仍不完整則回傳 503。
再次調整時，`original` 應使用最近一次成功回傳的完整內容。
本端點沒有重新搜尋 RAG，也不接受 `selected_long_goal` 或參考資料 ID 作為處理欄位。

## 6. 各階段實際使用的個案資料

| API | 實際提供給分析／生成流程的資料 |
|---|---|
| `/v1/context` | 回傳基本資料、現況摘要、夢想 |
| `/v1/long-term-goals` | 基本資料、現況摘要、夢想、自填對他／為他重要 |
| `/v1/references/search` | 領域、成人類型、基本資料、選定長程目標；摘要與夢想雖被讀取，但沒有直接帶入搜尋 planner |
| `/v1/goals/generate` | 領域、成人類型、基本資料、選定長程目標及所採用參考資料；沒有直接帶入現況摘要與夢想 |
| `/v1/goals/refine` | 領域、成人類型、完整原結果與調整方向；讀取個案資料用於確認可存取，但沒有將摘要與夢想加入修改 prompt |

所有業務 API 都是讀取／生成，不會將 AI 結果寫回 Firestore。
若使用者要保存內容，前端應走原有的 ISP 儲存流程與權限控制。

## 7. 錯誤格式與限制

應用程式錯誤範例：

```json
{
  "detail": "登入憑證無效、已過期或已撤銷，請重新登入。",
  "request_id": "example-request-error"
}
```

| HTTP | 含義 | 前端建議 |
|---|---|---|
| 400 | 不支援的領域、無效或不符類型的參考 ID、資訊不足 | 修正輸入或重新搜尋參考資料 |
| 401 | Bearer Token 未提供、無效、過期、撤銷或帳號停用 | 更新 Token 一次；仍失敗時要求重新登入 |
| 403 | 非公司帳號、非在職員工、角色／服務／領域權限不足或個案不符條件 | 顯示無權操作提示，不自動重試 |
| 404 | 使用者、ISP 不存在或 ISP 已刪除 | 更新選單與選擇 |
| 413 | request body 超過 256 KiB | 縮短內容 |
| 422 | 欄位缺少、型別錯誤、長度超限或 `service_type` 不是成人 | 檢查 JSON，錯誤回應不包含逐欄位細節 |
| 429 | instance 內已驗證請求超過每分鐘 30 次 | 依 `Retry-After: 60` 等待，保留使用者輸入 |
| 500 | 未預期錯誤 | 顯示一般錯誤，提供 `request_id` 供查詢 |
| 503 | 員工驗證、Firestore、Chroma、模型、OpenAI 或重生成格式處理暫時失敗 | 保留輸入，讓使用者重試並提供 `request_id` |
| 504 | 雲端服務或中介層逾時 | 提示逾時；無法確定生成是否已完成 |

Cloud Run 自己產生的錯誤，可能是純文字或 HTML，不一定符合以上 JSON 格式。
連線／CORS 失敗也可能沒有可讀取的 HTTP 回應。

限流目前是每個 instance 的記憶體計數，所有已驗證員工請求共用，並非每個員工各有 30 次；不是全服務固定的全域額度。
實際同時在線人數不等於正在生成的請求數。最多 5 instances 是擴展設定，不是完成時間或精確成本上限保證。
閒置後第一個請求可能有冷啟動等待。

## 8. 前端狀態與逾時處理

- 長時間 API 呼叫顯示 loading，處理中停用同一個生成按鈕，避免重複請求。
- 直接呼叫 Cloud Run 時，可將 client timeout 設為 330 秒；部署的後端 timeout 為 300 秒。
- 每次呼叫時透過 `getIdToken()` 取得有效 Token；遇到 401 可 `getIdToken(true)` 更新並重試一次。不要對 403 或 503 套用更新 Token 的循環重試。
- 沒有 idempotency key 或結果查詢端點。`X-Request-ID` 只用於追蹤，不會阻止重複生成。
- 不建議對生成 API 自動無限制重試；重送可能再次呼叫 OpenAI 並產生費用。
- 瀏覽器取消請求不保證後端停止生成。
- 切換 student / ISP / domain 時，以選擇版本號或請求取消機制避免舊結果覆蓋新畫面。
- `result` 保留換行呈現；如解析七大面向，需處理格式不完整的情況。
- 不把完整個案內容、生成內容或金鑰記錄在 console／分析服務。

## 9. 瀏覽器呼叫範例

以下使用既有 Firebase 登入狀態，不建立另一套登入。

```typescript
import { getAuth } from 'firebase/auth';

const user = getAuth().currentUser;
if (!user) throw new Error('請先登入公司員工帳號');
const token = await user.getIdToken();
const response = await fetch(
  'https://mumucare-llm-663386326083.asia-east1.run.app/v1/long-term-goals',
  {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      studentId: 'student-example-001',
      ispId: 'isp-example-001',
      domain: '身體福祉',
      service_type: '成人',
      custom_important_to: [],
      custom_important_for: [],
    }),
  },
);
const body = await response.json();
if (!response.ok) throw new Error(`${body.detail} (${body.request_id})`);
```

Angular 使用專案既有注入的 Firebase Auth instance 取得 `currentUser`；不要初始化不同的 Firebase project。
HTTP interceptor 只應將 Token 附加到此 AI API 的 Base URL，不要附加到所有第三方網址。
請使用 Authorization header，不需設定 `withCredentials: true` 或 `credentials: 'include'`；此服務不採用 cookie 登入。

管理工具範例，`FIREBASE_ID_TOKEN` 需為真正登入後取得的短效憑證：

```bash
curl --request POST \
  'https://mumucare-llm-663386326083.asia-east1.run.app/v1/long-term-goals' \
  --header "Authorization: Bearer ${FIREBASE_ID_TOKEN}" \
  --header 'Content-Type: application/json' \
  --header 'X-Request-ID: frontend-example-001' \
  --data '{
    "studentId": "student-example-001",
    "ispId": "isp-example-001",
    "domain": "身體福祉",
    "service_type": "成人",
    "custom_important_to": [],
    "custom_important_for": []
  }'
```

同目錄的 `openapi.json` 提供完整 schema，可匯入 Postman 或用於產生型別。
範例 ID 需換成真正的 document ID。Firestore 讀取已驗證通過；若收到 503，請保留 `request_id` 供後端查詢，不應一律判斷為權限問題。
