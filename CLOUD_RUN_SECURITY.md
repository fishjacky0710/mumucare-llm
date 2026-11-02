# Cloud Run 安全部署

後端正式環境採用以下基線：

- `/v1/*` 強制驗證 `Authorization: Bearer <Firebase ID Token>`，不再接受共用 API key。
- Firebase Auth 專案為 `mumu-care`；驗證簽章、期限、撤銷、停用、公司信箱與 Google 登入 provider。
- 查詢 `users/{uid}`、在職 `employees` 與 `roles`，並以服務單位及成人領域角色授權。
- OpenAI key 由 Secret Manager 注入，不放在 image 或部署指令中。
- Cloud Run 使用專用服務帳號，資料專案授予 `roles/datastore.viewer` 與 `roles/firebaseauth.viewer`，均為唯讀。
- 正式環境只允許明確的 HTTPS CORS origin；未設定時不允許瀏覽器跨來源存取，且關閉 Swagger、ReDoc 與 OpenAPI schema。
- 單一 instance 併發設為 1、最多 5 個 instances，每 instance 每分鐘最多 30 個已驗證請求。
- 使用按請求計費（`--cpu-throttling`），最少 instances 為 0；閒置時不收取 Cloud Run CPU 與記憶體費用。啟動、處理請求及關閉仍會計費，映像儲存與 Secret Manager 等服務另計。
- request body 上限 256 KiB；回應包含防快取、禁止 MIME sniffing、禁止 iframe 等安全標頭。
- 健康檢查只回傳服務狀態，不揭露模型、金鑰與資料庫設定。

## 首次設定

`.env` 已由 git 與 Docker build context 排除。先設定：

```dotenv
GCP_PROJECT_ID=your-project-id
FIRESTORE_PROJECT_ID=mumu-care
FIREBASE_AUTH_PROJECT_ID=mumu-care
EMPLOYEE_EMAIL_DOMAIN=maria.org.tw
OPENAI_API_KEY=your-openai-api-key
CORS_ALLOW_ORIGINS=https://your-frontend.example.com
```

接著執行：

```bash
chmod +x setup_cloud_security.sh deploy.sh
./setup_cloud_security.sh
./deploy.sh
```

前端沿用既有 Firebase 登入，使用 `currentUser.getIdToken()` 取得員工憑證。直接呼叫 Cloud Run，不需要 Gateway 或共用 API key。

後端映像儲存於部署專案的 Artifact Registry。Firestore 由 `FIRESTORE_PROJECT_ID` 指定；執行服務帳號在資料專案需具備 `roles/datastore.viewer`。Firebase Auth 由 `FIREBASE_AUTH_PROJECT_ID` 指定；撤銷／停用檢查需 `roles/firebaseauth.viewer`。若部署人員無法修改 IAM，可用 `CONFIGURE_FIRESTORE_IAM=false` 執行初始化，並請專案管理員另行授權。權限不足時回傳 503，不跳過驗證。

Cloud Build、Artifact Registry 與 Cloud Run 使用同一個 `GCP_REGION`，預設為 `asia-east1`。部署腳本明確指定建置區域，並使用 regional user-owned buckets，讓後續建置的原始碼暫存與日誌儲存桶也使用該區域；既有 global 建置記錄與舊儲存桶不會搬移或刪除。後續建置請使用 `./deploy.sh`，手動呼叫 `gcloud builds submit` 時也需帶上相同區域與儲存桶參數。

部署腳本預設允許 `https://mumu-care.web.app`、`https://mumu-care.firebaseapp.com` 與測試網址 `https://mumu-care--jacky-lryp8mk8.web.app`。其他正式或測試網域需明確配置 `CORS_ALLOW_ORIGINS`，不使用萬用來源。

## 金鑰輪替

1. 在 `.env` 更新 `OPENAI_API_KEY`。
2. 再執行 `./setup_cloud_security.sh`，建立新的 secret version。
3. 執行 `./deploy.sh` 建立新 revision，前端不用接觸這把 key。
4. 確認新版本正常後，停用 Secret Manager 中的舊版本。

## 上線後

- 在 Google Cloud Billing 建立預算與用量告警。
- 在 OpenAI 專案設定月預算與速率限制。
- 定期檢查 Cloud Run 401、413、429、5xx 日誌及異常流量。
- Cloud Run ingress 可公開連線，業務 API 由 Firebase 員工登入與服務／領域權限保護；匿名只可讀取基本服務狀態。
- 舊版 `mumucare-api-key` secret 不再注入新 revision；若無其他服務使用，可由管理員停用舊 secret version。
