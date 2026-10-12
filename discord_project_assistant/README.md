# Discord 專案助理 Bot

本 Bot 以唯讀方式索引 Basketball-51 repository 內的可信文字文件，提供繁體中文 slash commands。它不執行文件或使用者提供的命令／程式碼，不開啟影片、不載入 checkpoint，也不執行訓練、資料清理或 evaluation。

## 功能與命令

| 指令 | 行為 | AI 未設定時 |
| --- | --- | --- |
| `/status` | 目前完成、進行中與研究範圍 | 本機檢索可用 |
| `/next` | 下一步、阻塞與原因 | 本機檢索可用 |
| `/why <question>` | 解釋文件記載的決策原因 | 降級為本機檢索 |
| `/ask <question>` | 自然語言專案問答 | 降級為本機檢索 |
| `/experiments` | A1/A2/B/C 與後續研究的重要結果／限制 | 本機檢索可用 |
| `/sources` | 該使用者最近一次答案的來源 | 可用 |
| `/refresh` | 僅重建有變更的 allowlisted 文件；管理員限定 | 可用 |
| `/health` | Bot、索引、AI 設定與當日用量 | 可用 |

每個答案都列出 `path — heading (line range)`。AI prompt 要求區分「文件明確記載」、「根據證據推論」及「資料不足」。檢索到狀態衝突時會列出文件日期並以較新文件作目前狀態；舊文件仍保留為歷史證據。

## 資料流與信任邊界

1. `config/project_sources.yaml` 是唯一檔案 allowlist。
2. loader 拒絕 path traversal、所有 symlink、repository 外路徑、secret-like 檔名／內容、binary、非允許副檔名、單檔／總量超限。
3. Markdown 依標題與大小切片；每片保存相對路徑、標題、行號、mtime、SHA-256 與文件日期。
4. SQLite 保存本機 BM25 類 ranking 所需 token counts。`/refresh` 以內容 hash 判斷新增、更新、未變與移除。
5. 只把排名最高且在 `MAX_CONTEXT_CHARS` 內的片段交給 AI。repository 不可寫；SQLite 位於 repository 外的 `BOT_DATA_DIR`。
6. AI 使用部署帳號在專用 `CODEX_HOME` 中快取的 ChatGPT OAuth 登入，啟動本機 `codex app-server` stdio session；不讀取或接受 OpenAI API key。每次回答使用 ephemeral thread 與空白暫存工作目錄，read-only sandbox 的可讀根只包含該暫存目錄；repository 內容只能經過已檢索的 prompt 片段進入模型。
7. App Server 的 MCP、web search、login shell 與 approval escalation 關閉。若事件流出現 command、file change、MCP、dynamic/collaboration tool、web search、image view 或任何 server-initiated request，Bot 立即丟棄 AI 結果並回退本機檢索。文件中的 prompt、命令與角色要求全部視為不可信資料。

目前未直接索引 `.git` internals，避免擴大讀取面與 runtime command execution。Git 決策歷程以已追蹤的 `docs/progress_log.md`、各 protocol 與報告為可查來源；若團隊未來需要完整 commit metadata，應先產生經審閱、無敏感內容的 tracked snapshot，再明確加入 allowlist。

## Discord Developer Portal

1. 在 [Discord Developer Portal](https://discord.com/developers/applications) 建立 Application，於 **Bot** 頁建立 Bot，複製 token 到部署機的 `.env`；不要貼到 Discord 或 commit。
2. 本程式只用 slash commands，`discord.Intents.none()`，不需要 Message Content、Presence 或 Server Members privileged intents。
3. 在 **Installation**／OAuth2 URL Generator 選 **Guild Install**，scopes 選 `bot` 與 `applications.commands`。Discord 官方說明 `applications.commands` 會隨 `bot` scope 自動包含，但顯式勾選有助於檢查。此 Bot 不需要讀訊息或管理伺服器。
4. Bot permissions 只給 **View Channels** 與 **Send Messages**；不要給 Administrator、Manage Server、Manage Roles 或讀取歷史訊息。slash command 可用性可在 Server Settings → Integrations 進一步限制 role/channel。
5. 邀請至指定 server，開啟 Discord Developer Mode，複製 server、channel 與 role IDs，填入四個 allowlist 環境變數。
6. 首次啟動會為 `DISCORD_GUILD_IDS` 中的 server 同步 guild commands；guild commands 更新較快且不會出現在 DM。官方參考：[Application Commands](https://docs.discord.com/developers/interactions/application-commands)、[OAuth2](https://docs.discord.com/developers/topics/oauth2)。

應用程式本身會再次驗證 guild、channel 與 role；Discord UI 權限不是唯一安全層。`/refresh` 需要 `DISCORD_ADMIN_ROLE_IDS` 或 Discord Administrator 權限。

若要讓指定伺服器的所有成員使用一般指令，可將該伺服器的 Guild ID 填入 `DISCORD_ALLOWED_ROLE_IDS`；Discord 的 `@everyone` role ID 即 Guild ID。這個捷徑不適用於 `DISCORD_ADMIN_ROLE_IDS`，管理員仍必須使用明確的管理員 role ID 或具備 Discord Administrator 權限。

## 本機環境與 dry-run

以下命令都從 `basketball/discord_project_assistant/` 執行。現有研究環境是 Conda `yolo11`；為避免改變 frozen 研究依賴，建議建立獨立環境。若團隊決定沿用既有 Python 環境，也只安裝本資料夾的 `requirements.txt`：

```bash
conda create -n basketball-bot python=3.13 -y
conda activate basketball-bot
python -m pip install -r requirements.txt
```

主機還需要官方 Codex CLI；Docker image 會安裝固定版本，本機可先確認：

```bash
codex --version
```

不需要任何真實憑證的 smoke test：

```bash
PROJECT_ROOT="$(cd .. && pwd -P)" SOURCES_CONFIG="$PWD/config/project_sources.yaml" \
AI_PROVIDER=local python -m project_assistant_bot --dry-run --question "目前完成了什麼？"
```

dry-run 使用暫存 SQLite，強制關閉 AI，結束即移除，不寫 repository。正式本機啟動前，複製 `.env.example` 為受保護的 `.env` 並載入環境變數；程式不會自行解析或記錄 `.env`。

```bash
set -a
. ./.env
set +a
python -m project_assistant_bot --refresh-only
python -m project_assistant_bot
```

## Docker 部署

1. 複製 `.env.example` 為 `.env`，只填入 Discord token／allowlists；不要填入 Platform API key 或 Codex access token。
2. `docker compose build`。
3. `docker compose run --rm project-assistant-bot python -m project_assistant_bot --dry-run`。
4. 先以同一 named volume 執行 `docker compose run --rm project-assistant-bot codex -c 'forced_login_method="chatgpt"' -c 'cli_auth_credentials_store="file"' login status`。若已登入，直接進到步驟 7。
5. 若 workspace 未開放 device-code login，在 Linux 主機使用一次性的 host-network override 完成一般瀏覽器登入：

   ```bash
   docker compose -f compose.yaml -f compose.login.yaml run --rm \
     project-assistant-bot \
     codex \
     -c 'forced_login_method="chatgpt"' \
     -c 'cli_auth_credentials_store="file"' \
     login
   ```

   這個 override 只用於登入。Codex CLI 的 callback 監聽在 loopback，單純發布 `127.0.0.1:1455` 可能無法從 Docker bridge 到達容器 loopback。不要把 OAuth URL、callback URL、授權碼或 `/data/codex-home/auth.json` 貼到聊天或 log。
6. 再執行步驟 4 的 `login status`，確認登入成功。
7. `docker compose up -d`。
8. `docker compose logs -f --tail=200 project-assistant-bot`。

Compose 將專案掛載為 `/repo:ro`，root filesystem 也是 read-only，移除 Linux capabilities，設定 `no-new-privileges`，只允許 named volume `/data` 與小型 `/tmp` 寫入。`restart: unless-stopped` 讓主機重開機後在 Docker service 啟動時恢復。

Image 依 OpenAI 的 Linux sandbox 指引安裝發行版 `bubblewrap`。Codex 的 Linux read-only sandbox 仍需要主機允許安全的 unprivileged user namespaces；部分 Docker／受管主機會禁止 nested user namespaces。本程式偵測到 sandbox warning 時會拒絕 AI turn 並回退本機檢索，絕不改成 unsandboxed execution。若重新建置後 Docker `/health` 仍顯示 sandbox 不可用，建議改用下方 systemd 部署，或保持 `AI_PROVIDER=local`；不要加入 `--privileged`、`SYS_ADMIN` 或關閉 seccomp 來繞過。官方說明：[Codex sandbox](https://learn.chatgpt.com/docs/sandboxing)。

## systemd 部署

`deploy/basketball-project-bot.service` 是範本。管理員需：

1. 安裝 OpenAI 官方 Linux sandbox 文件建議的 `bubblewrap` 與系統 Python 的 venv 套件，再建立無登入權限的 `basketball-bot` 使用者。Ubuntu 可執行 `sudo apt install bubblewrap python3-venv`；若目前的系統 Python 小版本較新且 venv 內沒有 pip，安裝對應套件，例如 `python3.14-venv`。
2. 僅授予服務帳號穿越使用者 home 與讀取 repository 的 ACL；不要讓它讀取 `.env`、`.ssh` 或整個 home：

   ```bash
   sudo setfacl -m u:basketball-bot:--x /home/tsai
   sudo setfacl -R -m u:basketball-bot:rX /home/tsai/projects/basketball
   sudo setfacl -R -m u:basketball-bot:--- \
     /home/tsai/projects/basketball/discord_project_assistant/.env
   ```

3. 在 `/opt/basketball-project-bot-venv` 建立 venv 並安裝本資料夾的 `requirements.txt`。
4. 安裝官方 Codex CLI，確認 `/usr/local/bin/codex --version`；若路徑不同，修改 unit 的 `CODEX_BINARY`。
5. 建立只有服務帳號能進入的狀態目錄：

   ```bash
   sudo install -d -o basketball-bot -g basketball-bot -m 0700 \
     /var/lib/basketball-project-bot/codex-home
   ```

6. 由**同一個服務帳號**執行一次一般 ChatGPT 瀏覽器登入，並選擇你的 Business workspace。這是在主機 network namespace 執行，因此 `127.0.0.1:1455` callback 可直接到達 Codex CLI，不需要 device-code 權限：

   ```bash
   sudo -u basketball-bot env \
     HOME=/var/lib/basketball-project-bot \
     CODEX_HOME=/var/lib/basketball-project-bot/codex-home \
     /usr/local/bin/codex \
     -c 'forced_login_method="chatgpt"' \
     -c 'cli_auth_credentials_store="file"' \
     login
   ```

7. 不顯示憑證內容，只檢查登入狀態：

   ```bash
   sudo -u basketball-bot env \
     HOME=/var/lib/basketball-project-bot \
     CODEX_HOME=/var/lib/basketball-project-bot/codex-home \
     /usr/local/bin/codex \
     -c 'forced_login_method="chatgpt"' \
     -c 'cli_auth_credentials_store="file"' \
     login status
   ```

8. 確認 `/var/lib/basketball-project-bot/codex-home/auth.json`（若產生）只允許 `basketball-bot` 讀取；不得複製到 repository、備份、工單或聊天。
9. 將 Discord token 與 allowlists 寫入 root-owned、mode `0600` 的 `/etc/basketball-project-bot.env`；不放任何 OpenAI API key 或 Codex access token。
10. 複製 unit 至 `/etc/systemd/system/`，檢查 repository 路徑與使用者。
11. 啟動 systemd 前先執行 `docker compose down`；同一 Discord token 不應同時跑兩個 Bot instance。
12. `sudo systemctl daemon-reload && sudo systemctl enable --now basketball-project-bot`。
13. `sudo journalctl -u basketball-project-bot -n 100 --no-pager`，再於允許的 Discord channel 執行 `/health`；應顯示 `ChatGPT 登入快取有效`。

unit 使用 `ReadOnlyPaths`、獨立 `StateDirectory`、`UMask=0077` 與 systemd hardening；log 由 journald 管理。ChatGPT 登入快取在 `CODEX_HOME`，不在環境檔、unit、command line 或 Git 中。官方說明 `CODEX_HOME` 同時供 CLI 與 app-server 保存 auth/state，因此登入與服務必須使用完全相同的路徑：[Environment variables](https://learn.chatgpt.com/docs/config-file/environment-variables)。

`RestrictAddressFamilies` 必須包含 `AF_NETLINK`，供 Bubblewrap 在自己的隔離 network namespace 中設定 loopback；這不會覆寫 Bot 的 permission profile，command network 仍為 disabled。修改 unit 後必須重新複製到 `/etc/systemd/system/` 並執行 `systemctl daemon-reload`，單純 restart 不會載入 repository 內的新 unit 範本。

啟用 AI 前先以部署使用者確認 `codex app-server` 的 Linux sandbox 沒有 user-namespace／bubblewrap 警告。Bot 使用 Codex permission profile（需 Codex 0.138.0 以上；目前測試版本為 0.160.0），全域拒絕檔案存取，只開放最小執行環境與每次回答建立的空白暫存 workspace 唯讀存取，並停用 command network。它不使用已淘汰的 `readOnly.access`。若主機政策禁止該 sandbox，Bot 仍可部署，但 AI 會 fail closed；請勿降低 OS 隔離設定。官方說明：[Permission profiles](https://learn.chatgpt.com/docs/permissions)、[App Server experimental API](https://learn.chatgpt.com/docs/app-server)。

## 索引更新、備份與 log

- 啟動會 refresh 一次；管理員可用 `/refresh`。也可在維護時執行 `python -m project_assistant_bot --refresh-only`。
- refresh 會讀取 allowlisted 檔案並計算 hash；只有內容變更的文件重建 chunks。mtime 改變但內容相同不重建內容。
- 備份只需要停止 Bot 後複製 `BOT_DATA_DIR/project_index.sqlite3`；它可由 repository 重建，因此不應與 source backup 混為正式研究證據。
- Docker named volume 可用組織既有的 volume backup 方法；systemd 預設在 `/var/lib/basketball-project-bot`。
- log 只記 lifecycle、計數與錯誤類型，不記 token、完整 prompt 或文件內容。不要把 debug log 公開。

## ChatGPT 登入、Business 額度與成本保護

本版本不使用 OpenAI Platform API key、`CODEX_ACCESS_TOKEN` 或付費 API fallback。AI 路徑沿用部署帳號透過 `codex login` 建立的 ChatGPT OAuth 登入快取；app-server 與 CLI 使用相同 credentials。官方文件說明 ChatGPT 登入使用訂閱方案權益、登入可由 CLI 快取並在使用期間自動更新：[Authentication](https://learn.chatgpt.com/docs/auth)、[Codex CLI reference](https://learn.chatgpt.com/docs/cli/reference)。

### 你需要先完成的 Business／登入設定

1. 請 workspace owner 允許你的帳號使用 Codex Local；若要用 device code，還需在 workspace permissions 允許 device code login。
2. 依 systemd 步驟，以 `basketball-bot` 帳號執行 `codex login --device-auth`，登入你自己的 ChatGPT 帳號並選擇正確 Business workspace。
3. `forced_login_method="chatgpt"` 防止誤用 API-key 登入；程式也不會把父程序的 `OPENAI_API_KEY`、`CODEX_ACCESS_TOKEN` 或 Discord token 傳給 app-server。
4. 在 Business Billing／Usage controls 將 workspace overage limit 設為 `0`；若方案介面提供 individual override，再替你的帳號設定個別上限。警示不是硬上限。官方說明 overage 為零時，用完可用 credits 後功能會暫停：[ChatGPT Work usage and cost](https://learn.chatgpt.com/docs/enterprise/chatgpt-work-usage-and-cost)。

重要限制：ChatGPT 登入會把執行歸屬到你選擇的 workspace identity，但 Business 若已有共享 workspace credits，消耗規則仍由 workspace 合約與管理設定決定。登入模式不代表免費或絕對只扣個人額度。程式可以保證不切換到 Platform API、額度錯誤時停止 AI，不能替代管理員的 shared-credit／overage 控制。所有獲准使用 `/ask`、`/why` 的 Discord 成員都會消耗這個登入身分的可用量。

可調整設定：

- `AI_PROVIDER=codex`；改為 `local` 可完全停用 AI。
- `CODEX_AUTH_MODE=chatgpt_login`：唯一支援的 AI 認證模式；其他值會 fail closed。
- `CODEX_MODEL=gpt-6-luna`、`CODEX_REASONING_EFFORT=low`：預設採節省額度的組合；模型不可用時會 fail closed。
- `CODEX_TIMEOUT_SECONDS=45`、`AI_MAX_ANSWER_CHARS=3500`：限制等待與回覆長度。
- `MAX_CONTEXT_CHUNKS=8`、`MAX_CONTEXT_CHARS=16000`：限制交給模型的檢索片段。
- `DAILY_AI_REQUEST_LIMIT=40`、`PER_USER_DAILY_AI_REQUEST_LIMIT=5`：SQLite 原子計數；失敗請求也保守占用一次額度。
- `PER_USER_MINUTE_COMMAND_LIMIT=12`：所有指令的 process-local burst protection。

尚未登入、CLI 不存在、timeout、`UsageLimitExceeded`、登入失效、provider error 或工具嘗試時，Bot 都回退本機檢索。`/status`、`/next`、`/experiments` 固定不用 AI。`/health` 會透過 `codex login status` 回報登入是否存在與最近錯誤類型，不顯示任何 OAuth credential。

### 為何不在 Discord Bot 內自行實作 OAuth callback

這個 Bot 是單一主機、單一受信任登入身分的 systemd daemon；直接使用官方 Codex CLI device login 即可，不需要在 Bot 內增加 OAuth callback、PKCE、Discord identity linking 或 per-user token vault。禁止拷貝 ChatGPT browser cookies、使用非官方 token exchange 或繞過 workspace 授權。

## 安全與不可推論事項

- `.env`、credentials、tokens、SSH keys、影片、checkpoint、manifest、原始資料、大型 artifacts 與未 allowlist 檔案永不進索引。
- `.env.example` 只有 placeholder；真實 secrets 必須放在部署平台的 secret store 或受保護環境檔。
- Bot 的 repository mount 必須唯讀；請勿把 `BOT_DATA_DIR` 設在 repository 內。
- allowlist 變更視為 security review：先檢查資料分類、大小、是否含個資／secret 與 prompt injection，再部署。
- 檢索與 AI 回答不能取代原始 protocol／report。完整影片結果不能證明 pre-release 能力；A1/A2/B/C 是完整流程比較，不能宣稱單變因因果。
- `templates/PROJECT_STATUS.example.md` 是空白範本，整個 `discord_project_assistant/` 都在 deny list，避免操作文件或示例文字被當作研究事實。

## 疑難排解

- **slash commands 看不到**：確認 Bot 已以 Guild Install 邀請、guild ID 正確、`DISCORD_SYNC_COMMANDS=true`，再看 `/health` 或 startup log。
- **顯示未授權**：確認實際互動所在 guild/channel，以及使用者 role ID 都在環境變數；ID 必須是純數字、逗號分隔。
- **索引 0 文件**：確認 `PROJECT_ROOT`、`SOURCES_CONFIG` 與 read-only mount；dry-run 會列出 scanned/skipped 數量。
- **文件被略過**：檢查 symlink、UTF-8、secret pattern、檔案大小與 allowlist。不要為了通過索引而移除安全檢查；先縮小或製作經審閱摘要。
- **AI 不可用**：`/health` 顯示登入狀態與最近失敗類型；以 `basketball-bot` 帳號和相同 `CODEX_HOME` 執行 `codex login status`，再檢查 Business Local/device-code 權限、model access 與 usage limit。核心命令仍可用。
- **`readOnly.access is no longer supported`**：執行中的程式碼仍是舊版 provider。更新本資料夾後重新啟動服務；新版改用 Codex permission profile，不再傳送 `readOnly.access`。systemd 執行 `sudo systemctl restart basketball-project-bot`；Docker 必須 `docker compose up -d --build`，只 restart 不會更新 image。
- **`bwrap: loopback: Failed to create NETLINK`**：systemd 的 address-family allowlist 缺少 `AF_NETLINK`。更新安裝中的 unit，使 `RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK`，再執行 `sudo systemctl daemon-reload` 與 restart。不要改成 `danger-full-access` 或停用 sandbox。
- **Device login 不可用**：Docker/Linux 可依部署章節使用 `compose.login.yaml` 的 host-network override，以相同 named volume 執行一般 `codex login`。若一般瀏覽器登入也被 workspace policy 拒絕，必須請 workspace owner 啟用 Codex Local／相關登入權限，不得繞過。不要複製瀏覽器 cookie。
- **UsageLimitExceeded**：這是預期的 fail-closed 行為；Bot 會繼續提供本機檢索。不要為此填入 Platform API key，等方案額度重置或由管理員調整既有 Business 控制。
- **database is locked**：只執行一個 Bot instance；備份前停止 service，不要在共享網路檔案系統同時寫同一 SQLite。
- **回答引用舊狀態**：執行 `/refresh`，檢查文件日期與 `current_research_direction.md`；若來源本身衝突，先修正文檔治理，不要讓 Bot 猜測。
